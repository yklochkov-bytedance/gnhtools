import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor
from torch.distributed.tensor.parallel._utils import (
    _prepare_input_validate,
    _prepare_output_validate,
)

from typing import Union

@_prepare_input_validate
def _local_tensors_to_sharded(
    local_tensor: Union[DTensor, torch.Tensor],
    device_mesh: DeviceMesh
    ) -> Union[DTensor, torch.Tensor]:
    """
        Allows to turn a local torch.Tensor "local_tensor", each on it's rank into a sharded DTensor.

        For example, in attention modules, we calculate teh attention weights with local keys, queries, and values.
        Each rank has its own portion of heads. Once the attention outputs are calculated,we collect them into a 
        sharded DTensor and pass it to the output projection layer.
    """
    dtensor = DTensor.from_local(local_tensor, device_mesh, [Shard(2)], run_check=False)
    return dtensor

@_prepare_output_validate
def _to_local_tensor(
    dtensor: DTensor,
    device_mesh: DeviceMesh
    ) -> torch.Tensor:
    """
        This funciton turns the input DTensor into a local tensor, each corresponding to its rank.

        We typically use it with sharded tensors, when we want to conduct operations with each shard independently
        on each rank.

        It can also be useful to turn a replicated DTensor to local tensor, which are essentially equivalent, but
        some of the operations are not implemented for DTensors. For example, we can turn a replicated DTensor to 
        local, apply a SiLU activation function, then turn it back to a replicated Tensor. Without this step we would
        get an error `opertaion not implemented`.
    """
    return dtensor.to_local()



# helper parallelizing functions for linear modules

def _colwise_parallelize_conv1d(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the module :class:`transformers.pytorch_utils.Conv1d` module in :class:`ColwiseParallel`
    style. Conv1D conducts matrix multiplication by module.weight, so we need to shard the weight matrix by the 1st 
    dimension (starts with 0).

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`Conv1D` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """

    for name, param in module.named_parameters():
        if name.endswith("weight"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(1)])
            )
        elif name.endswith("bias"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
        else:
            raise ValueError("Expect only weight and bias parameters.")
        module.register_parameter(name, dist_param)

def _rowwise_parallelize_conv1d(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the module :class:`transformers.pytorch_utils.Conv1d` module in :class:`ColwiseParallel`
    style. Conv1D conducts matrix multiplication by module.weight, so we need to shard the weight matrix by the 0th 
    dimension.

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`Conv1D` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """

    for name, param in module.named_parameters():
        if name.endswith("weight"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
        elif name.endswith("bias"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Replicate()])
            )
        else:
            raise ValueError("Expect only weight and bias parameters.")
        module.register_parameter(name, dist_param)


def _colwise_parallelize_linear(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the module :class:`nn.Linear` module in :class:`ColwiseParallel` style. Linear layer
    conducts matrix multiplication by module.weight.T, so we need to shard the weight matrix by the 0th dimension.

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """

    for name, param in module.named_parameters():
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, [Shard(0)])
        )
        module.register_parameter(name, dist_param)


def _rowwise_parallelize_linear(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the module :class:`nn.Linear` module in :class:`ColwiseParallel` style. Linear layer
    conducts matrix multiplication by module.weight.T, so we need to shard the weight matrix by the 1st dimension
    (starts with 0).

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """

    for name, param in module.named_parameters():
        if name.endswith("weight"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(1)])
            )
        elif name.endswith("bias"):
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Replicate()])
            )
        else:
            raise ValueError("Expect only weight and bias parameters.")
        module.register_parameter(name, dist_param)
