"""
    This file contains classes that we use to iterate over models parameters, their gradients,
    calculating their dot products, projecting them, etc.
    We also leave out an option to create a container, so we treat it as a vector,
    which has the same shape as a set of parameters of a model. This is required to avoid
    vectorizing the whole model.  
"""

import torch
import torch.distributed as dist

from math import sqrt
import numpy as np

from torch.distributed._tensor import DTensor, Replicate, Shard
import tp_utils

def _safe_norm(tensor):
    if tensor.dtype is torch.float32:
        return tensor.norm()

    i = int(np.argmax(tensor.size()))
    return tensor.norm(dim=i).to(torch.float32).norm() #.to(tensor.dtype)

def _distributed_safe_norm(dtensor):
    """
        This function is used to avoid overflows when calculating the norm of a distributed tensor or local tensor.

        In particular, we are concerned with overflows for float16

        Input:
            tensor: Union[torch.Tensor, DTensor]
        Output:
            value: torch.Tensor (scalar)
    """

    if type(dtensor) is torch.Tensor:
        local_tensor = dtensor
    elif type(dtensor) is DTensor:
        local_tensor = dtensor._local_tensor
    
    #norm = _safe_norm(local_tensor).unsqueeze(0)
    norm = local_tensor.norm().unsqueeze(0)
    if type(dtensor) is DTensor:
        if type(dtensor.placements[0]) is Shard:
            norm = DTensor.from_local(norm, dtensor.device_mesh, [Shard(0)])
            norm = norm.redistribute(dtensor.device_mesh, [Replicate()])
            norm = norm.to_local()

    max_norm = norm.max()
    norm = norm / max_norm # avoid large numbers before taking squares
    norm = max_norm * norm.square().sum().sqrt()
    return norm

def _get_local(t):
    """
        Returns local tensor of t (DTensor or torch.Tensor), and a bool indicating whether
        the tensor is sharded or not
    """
    if isinstance(t, DTensor):
        is_sharded = isinstance(t.placements[0], Shard)
        return t._local_tensor, is_sharded
    else:
        return t, False


class BaseIterator:
    """
        Iterators are intended as vectorized versions of parameters of a model. Instead of vectorizing 
        explicitly, we calculate the dot products and norms in a streaming way, i.e., through iterating 
        over the correponding parameters.

        We implement a series of iterators. Each iterates over some tensors with shapes matching 
        parameters of a model, such as iteration over parameters themselves, iteration over gradients;
        a contrainer with tensors matching the size of parameters; random gaussian vector as a set of random
        tensors matching shapes of model's parameters, each drawn independently. Finally, a random embedding, 
        which is a series of normalized random vectors to serve as almost dot-product preseving embedding.
    """

    def __iter__(self):
        raise NotImplementedError()

    def dot(self, other):
        """
            Returns local torch.Tensor, replicated on all ranks
        """
        res = 0.0
        for t1, t2 in zip(self, other):
            t1_, is_sharded = _get_local(t1)
            t2_, _ = _get_local(t2)
            dot = torch.tensordot(t1_, t2_, dims = len(t1.size()))
            if is_sharded or tp_utils.get_rank() == 0:
                res += dot
        if self.device_mesh is not None:
            dist.all_reduce(res, op=dist.ReduceOp.SUM) # TODO: move to tp_utils for reproducibility without parallezation
        return res

    def square_norm(self, float32=True):
        """
            Calculate sum of squares of all tensors in the iterator.

            For float32=True we output result in float32 to avoid overflow.
        """
        if float32:
            return self.norm().to(torch.float32) ** 2
        else:
            return self.norm() ** 2

    def norm(self):
        res = 0.0
        for t in self:
            norm = _distributed_safe_norm(t)
            res += norm.to(torch.float32) ** 2
        return torch.sqrt(res).to(t.dtype).squeeze()

def _get_params_device_mesh(params):
    for par in params:
        if isinstance(par.data, DTensor):
            return par.data.device_mesh
    return None

class ParamIterator(BaseIterator):
    """
        Iterates over detached parameters of a model.
    """

    def __init__(self, params, device_mesh=None):
        super().__init__()
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        self.device_mesh = device_mesh
        self.params = [par for par in params if par.requires_grad]

        if device_mesh is None:
            self.device_mesh = _get_params_device_mesh(self.params)

    def __iter__(self):
        return map(lambda par: par.data, self.params)


class GradIterator(BaseIterator):
    """
        Iterates over gradients of the parameters of a model.
    """

    def __init__(self, params, device_mesh=None):
        super().__init__()
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        self.params = [par for par in params if par.requires_grad]
        
        if device_mesh is None:
            self.device_mesh = _get_params_device_mesh(self.params)

    def __iter__(self):
        return map(lambda par: par.grad, self.params)


def _zeros_like(source_tensor):
    if type(source_tensor) is torch.Tensor:
        return torch.zeros_like(source_tensor)
    elif type(source_tensor) is DTensor:
        local_tensor = torch.zeros_like(source_tensor._local_tensor)
        tensor = DTensor.from_local(local_tensor, source_tensor.device_mesh, source_tensor.placements)
        return tensor
    else:
        ValueError(f"Tensor type not supported: {type(source_tensor)}")

class ContainerLike(BaseIterator):
    """
        Creates a place holder for a LiSSA result. For each parameter of a model, we create a tensor 
        with zeros with the same shape.

        Iterates over the corresponding placeholders.
    """

    def __init__(self, iter):
        super().__init__()
        self.tensors = [_zeros_like(t) for t in iter]
        self.device_mesh = iter.device_mesh

    def __iter__(self):
        """
            Notice that since we return the list, we can modify each of the tensors inplace,
            which is handy for LiSSA updates.
        """
        for t in self.tensors:
            yield t

    def add_(self, other, scale=None):
        for t, par in zip(self.tensors, other):
            if type(t) is DTensor:
                t = t._local_tensor
                par = par._local_tensor

            if scale is None:
                t += par
            else:
                t += scale * par

    def scale(self, val):
        for t in self.tensors:
            if type(t) is torch.Tensor:
                t *= val
            elif type(t) is DTensor:
                t._local_tensor *= val

    def copy_from(self, other):
        for t, par in zip(self.tensors, other):
            if type(t) is torch.Tensor:
                t[...] = par
            elif type(t) is DTensor:
                t._local_tensor[...] = par._local_tensor
            else:
                ValueError(f"Tensor type not supported: {type(t)}")


class RandomLike(BaseIterator):
    """
        Random vector with normalization N(0, 1/N I) where N is total dimension of parameter
    """

    def __init__(self, other, local_seed, scale=None):
        self.device_mesh = other.device_mesh
        self.other = list(other)
        self.local_seed = local_seed

        # All generation happens on local rank
        self.device = torch.device(tp_utils.get_rank())

        # extract total dimension
        total_dim = 0
        for tensor in self.other:
            dim = 1
            for d in tensor.size():
                dim *= d
            total_dim += dim

        if scale is None:
            self.scale = 1.0 / sqrt(total_dim)
        else:
            self.scale = scale

    def __iter__(self):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.local_seed)

        for source_tensor in self.other:
            if type(source_tensor) is DTensor:
                local_size = source_tensor._local_tensor.size()
            else:
                local_size = source_tensor.size()
            tensor = torch.randn(
                *list(local_size),
                generator=gen,
                device=self.device,
                dtype=source_tensor.dtype)
            tensor *= self.scale
            if type(source_tensor) is DTensor:
                if isinstance(source_tensor.placements[0], Replicate):
                    # we will not get to here without init par group
                    dist.broadcast(tensor, 0) 
                tensor = DTensor.from_local(tensor, self.device_mesh, source_tensor.placements)
            else:
                # TODO: why is it here??
                if self.device_mesh is not None:
                    dist.broadcast(tensor, 0)
                    tensor = DTensor.from_local(tensor, self.device_mesh, [Replicate()]).to_local()
            yield tensor


# TODO: join RandomProjectorLike and InflatedRandomProjectorLike
# to avoid code repetition

class RandomProjectorLike(BaseIterator):
    """
        Creates a RandomProjection for parameters of distributed model. Two methods are available:

        - get_vector(i) returns the column number i as a vector, so it can be used in combination
        with other functions, e.g., computing iHVP

        - project(other) applies the random embedding to vector 'other'
    """

    def __init__(self, dim, other, local_seeds):
        super().__init__()

        self.device_mesh = other.device_mesh
        self.other = list(other)
        self.local_seeds = local_seeds

        # local_seeds is an array of size chunks x [local_shards]
        self.num_chunks = len(self.local_seeds)
        self.num_local_shards = len(self.local_seeds[0])

        # All generation happens on local rank
        self.device = torch.device(tp_utils.get_rank())

        self.dim = dim

        assert (
            self.dim % self.num_chunks == 0
        ), "Dimension must be divisible by the number of chunks."

        self.dim_per_chunk = self.dim // self.num_chunks

        # scale is defined by the mapping dimension
        self.scale = 1 / sqrt(dim)

        # to know which chunk, or which index in the chunk to iterate
        # not to change outside of the class please
        self._which_chunk = None
        self._which_idx = None
        self._iterate_local = False

    def get_column(self, col_number):
        self._which_idx = col_number % self.dim_per_chunk
        self._which_chunk = col_number // self.dim_per_chunk
        self._iterate_local = False
        return self

    def __iter__(self):
        """
            In case _which_idx is not specified (None), iterates over the projector matrices corresponding
            to self._which_chunk part
            In case _which_idx is specified, returns the slice of the projector matrices correponding to that
            index whithin the
        """
        gens = []
        for seed in self.local_seeds[self._which_chunk]:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            gens.append(gen)

        for source_tensor in self.other:
            if type(source_tensor) is DTensor:
                local_size = source_tensor._local_tensor.size()
            else:
                local_size = source_tensor.size()
            local_size = [self.dim_per_chunk] + list(local_size)

            # if local tensor or replicated tensor,
            # generate everything from zeroth generator
            # do a blank pass for reamining generators for consistency
            #
            if (type(source_tensor) is torch.Tensor or (
                type(source_tensor) is DTensor and
                isinstance(source_tensor.placements[0], Replicate))
                ):
                #for i in range(self.num_local_shards-1, -1, -1):
                if tp_utils.get_rank() == 0:
                    local_tensor = torch.randn(
                        *local_size,
                        generator=gens[0],
                        device=self.device,
                        dtype=source_tensor.dtype)
                else:
                    local_tensor = torch.zeros(
                        *local_size,
                        device=self.device,
                        dtype=source_tensor.dtype)
                    # TODO: just use the first shard

            # when the tensor is sharded, we have to concatenate the generations
            # from each local shard. Sharded dimension needs to be extraceted
            # from source tensor, and also add +1 in case _which_idx is None,
            #
            else:
                placement = source_tensor.placements[0]
                assert(isinstance(placement, Shard))
                shard_dim = placement.dim + 1 # must be Shard(dim) at this point,
                                              # shall we add an assertion?

                # dimension to pass to each generator must be sharded
                local_size[shard_dim] = local_size[shard_dim] // self.num_local_shards

                local_tensor = torch.cat(
                    [
                        torch.randn(
                            *local_size,
                            generator=gen,
                            device=self.device,
                            dtype=source_tensor.dtype)
                        for gen in gens
                    ],
                    dim=shard_dim
                )

            if self._which_idx is not None:
                local_tensor = local_tensor[self._which_idx, ...]

            local_tensor *= self.scale

            if self._iterate_local:
                # for dot products and projection we only
                # do communication once at the end
                #
                yield local_tensor

            else:
                # at this point local_tensor matches local part of source_tensor
                # and we need to compose a matching distributed tensor
                #
                if type(source_tensor) is DTensor:
                    if isinstance(source_tensor.placements[0], Replicate):
                        dist.broadcast(local_tensor, 0)
                    placements = source_tensor.placements
                    if self._which_idx is None and isinstance(placements[0], Shard):
                        # take into account that one dimension will be added in front
                        placements = [Shard(placements[0].dim + 1)]
                    tensor = DTensor.from_local(local_tensor, self.device_mesh, placements)
                else:
                    # again, why it's replicating a replicated tensor?
                    # to make sure it's replicated again?
                    if self.device_mesh is not None:
                        dist.broadcast(local_tensor, 0)
                        tensor = DTensor.from_local(local_tensor, self.device_mesh, [Replicate()]).to_local() # TODO: this one can probably be omited...
                    else:
                        tensor = local_tensor
                yield tensor

    def project(self, other):
        self._which_idx = None
        self._iterate_local = True
        ress = []
        for chunk in range(self.num_chunks):
            self._which_chunk = chunk
            res = 0.0
            for local_proj, vec in zip(self, other):
                vec_local = vec
                if type(vec) is DTensor:
                    vec_local = vec._local_tensor
                term = torch.tensordot(local_proj, vec_local, dims=len(vec.size()))
                res += term

            ress.append(res)

        ress = torch.hstack(ress)
        # sum up all terms once
        if self.device_mesh is not None:
            dist.all_reduce(ress, op=dist.ReduceOp.SUM)
        return ress


class InflatedRandomProjectorLike(BaseIterator):
    """
        Creates a RandomProjection for parameters of distributed model. Two methods are available:

        - get_vector(i) returns the column number i as a vector, so it can be used in combination
        with other functions, e.g., computing iHVP

        - project(other) applies the random embedding to vector 'other'
    """

    def __init__(self, dim_per_param, other, local_seeds, split_per_device=False):
        super().__init__()

        self.device_mesh = other.device_mesh
        self.other = list(other)
        self.local_seeds = local_seeds

        # local_seeds is an array of size chunks x [local_shards]
        self.num_chunks = len(self.local_seeds)
        self.num_local_shards = len(self.local_seeds[0])

        # All generation happens on local rank
        self.device = torch.device(tp_utils.get_rank())

        self.dim_per_param = dim_per_param
        self.dim = dim_per_param * len(self.other)

        assert (
            self.dim_per_param % self.num_chunks == 0
        ), "Dimension must be divisible by the number of chunks."

        self.dim_per_chunk = self.dim_per_param // self.num_chunks

        # scale is defined by the mapping dimension
        self.scale = 1 / sqrt(dim_per_param)

        # to know which chunk, or which index in the chunk to iterate
        # not to change outside of the class please
        self._which_chunk = None
        self._which_idx = None
        self._which_param = None
        self._which_device = None # TODO: only change if split_per_device=True, tbd
        self._iterate_local = False

    def get_column(self, col_number):
        #self._which_device = TODO:
        n_params = len(self.other)
        self._which_chunk = col_number // (self.dim_per_chunk * n_params)
        col_number = col_number % (self.dim_per_chunk * n_params)
        self._which_param = col_number // self.dim_per_chunk
        self._which_idx = col_number % self.dim_per_chunk

        self._iterate_local = False
        return self

    def __iter__(self):
        """
            In case _which_idx is not specified (None), iterates over the projector matrices corresponding
            to self._which_chunk part
            In case _which_idx is specified, returns the slice of the projector matrices correponding to that
            index whithin the
        """
        gens = []
        for seed in self.local_seeds[self._which_chunk]:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            gens.append(gen)

        for tensor_idx, source_tensor in enumerate(self.other):
            if type(source_tensor) is DTensor:
                local_size = source_tensor._local_tensor.size()
            else:
                local_size = source_tensor.size()
            local_size = [self.dim_per_chunk] + list(local_size)

            # if local tensor or replicated tensor,
            # generate everything from zeroth generator
            # do a blank pass for reamining generators for consistency
            #
            if (type(source_tensor) is torch.Tensor or (
                type(source_tensor) is DTensor and
                isinstance(source_tensor.placements[0], Replicate))
                ):
                #for i in range(self.num_local_shards-1, -1, -1):
                if tp_utils.get_rank() == 0:
                    local_tensor = torch.randn(
                        *local_size,
                        generator=gens[0],
                        device=self.device,
                        dtype=source_tensor.dtype)
                else:
                    local_tensor = torch.zeros(
                        *local_size,
                        device=self.device,
                        dtype=source_tensor.dtype)
                    # TODO: just use the first shard

            # when the tensor is sharded, we have to concatenate the generations
            # from each local shard. Sharded dimension needs to be extraceted
            # from source tensor, and also add +1 in case _which_idx is None,
            #
            else:
                placement = source_tensor.placements[0]
                assert(isinstance(placement, Shard))
                shard_dim = placement.dim + 1 # must be Shard(dim) at this point,
                                              # shall we add an assertion?

                # dimension to pass to each generator must be sharded
                local_size[shard_dim] = local_size[shard_dim] // self.num_local_shards

                local_tensor = torch.cat(
                    [
                        torch.randn(
                            *local_size,
                            generator=gen,
                            device=self.device,
                            dtype=source_tensor.dtype)
                        for gen in gens
                    ],
                    dim=shard_dim
                )

            if self._which_idx is not None:
                local_tensor = local_tensor[self._which_idx, ...]
            if (self._which_param is not None and 
                tensor_idx != self._which_param
                ):
                # null out everything away from "_which_param"
                local_tensor = torch.zeros_like(local_tensor)

            local_tensor *= self.scale

            if self._iterate_local:
                # for dot products and projection we only
                # do communication once at the end
                #
                yield local_tensor

            else:
                # at this point local_tensor matches local part of source_tensor
                # and we need to compose a matching distributed tensor
                #
                if type(source_tensor) is DTensor:
                    if isinstance(source_tensor.placements[0], Replicate):
                        dist.broadcast(local_tensor, 0)
                    placements = source_tensor.placements
                    if self._which_idx is None and isinstance(placements[0], Shard):
                        # take into account that one dimension will be added in front
                        placements = [Shard(placements[0].dim + 1)]
                    tensor = DTensor.from_local(local_tensor, self.device_mesh, placements)
                else:
                    if self.device_mesh is not None:
                        dist.broadcast(local_tensor, 0)
                        # TODO: this one can probably be omited...
                        tensor = DTensor.from_local(local_tensor, self.device_mesh, [Replicate()]).to_local()
                    else:
                        tensor = local_tensor
                yield tensor

    def project(self, other):
        inflated = True
        self._which_idx = None
        self._iterate_local = True
        ress = []
        for chunk in range(self.num_chunks):
            self._which_chunk = chunk
            res = 0.0
            for local_proj, vec in zip(self, other):
                if inflated:
                    res = 0.0

                vec_local = vec
                if type(vec) is DTensor:
                    vec_local = vec._local_tensor
                term = torch.tensordot(local_proj, vec_local, dims=len(vec.size()))
                res += term

                if inflated:
                    ress.append(res)

        if not inflated:
            ress.append(res)

        ress = torch.hstack(ress)
        # sum up all terms once
        if self.device_mesh is not None:
            dist.all_reduce(ress, op=dist.ReduceOp.SUM) # TODO:
        return ress
