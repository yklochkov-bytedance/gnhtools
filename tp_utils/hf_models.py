import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed._tensor import distribute_module
from torch.distributed.tensor.parallel import PairwiseParallel

# linear layer used in GPT2
from transformers.pytorch_utils import Conv1D

# for model-specific layer objects
import transformers.models.gpt2.modeling_gpt2 as transformers_modeling_gpt2
import transformers.models.gptj.modeling_gptj as transformers_modeling_gptj
import transformers.models.mistral.modeling_mistral as transformers_modeling_mistral
import transformers.models.llama.modeling_llama as transformers_modeling_llama
import transformers.models.opt.modeling_opt as transformers_modeling_opt


from tp_utils.functions import (
    _colwise_parallelize_linear,
    _rowwise_parallelize_linear,
    _colwise_parallelize_conv1d,
    _rowwise_parallelize_conv1d,
    _to_local_tensor,
    _local_tensors_to_sharded
)

# GPT-2

def _parallelize_gpt2_attn(attn, device_mesh): # TODO:
    """
        Parallelization of GPT2Attention layer. The layer has 2 submodules, and 
        2 buffers: c_attn, c_proj, masked_bias, bias.

        For the buffers, we use a local copy of both buffers on each gpu.
        
        We shard everything head-wise, for which we have to rearrange the columns of
        the c_attn layer, then shard them columnwise, so that each GPUs get's its
        portion of keys, queries, and values. Correspondingly, we shard the c_proj
        rowwise. We modify the intermediate input and output functions in a way

        There are NO buffers in GPT2MLP.

        Input can be any DTensor, output is local tensor.

        Arguments:
            module: GPT2Attention,
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized GPT2Attention
    """

    # Check if heads can be split evenly over the GPUs
    world_size = dist.get_world_size()
    assert attn.num_heads % world_size == 0, (
        f"Number of heads ({attn.num_heads}) must be divisible by world size({dist.get_world_size()})"
    )

    # First we rearrange the c_attn weights, so that each shard contains
    # the triplet kqv correponding to its allocated heads
    #
    def _rearrange_tripled(mat, world_size, dim):
        N_times_3 = mat.size()[-1]
        N = N_times_3 // 3
        assert(N_times_3 % (3 * world_size) == 0)

        # Stack the sub-matrices vertically
        result = torch.split(mat, N // world_size, dim=dim)
        idx = []
        for i in range(world_size):
            idx += [i, i + world_size, i + 2 * world_size]
        result = [result[j] for j in idx]
        return torch.cat(result, dim=dim)

    attn.c_attn.weight.data = _rearrange_tripled(attn.c_attn.weight.data, world_size, 1)
    attn.c_attn.bias.data = _rearrange_tripled(attn.c_attn.bias.data, world_size, 0)

    attn.num_heads = attn.num_heads // world_size
    attn.split_size = attn.split_size // world_size

    parallel_style = PairwiseParallel()
    distribute_module(
        attn.c_attn,
        device_mesh,
        _colwise_parallelize_conv1d,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.c_proj,
        device_mesh,
        _rowwise_parallelize_conv1d,
        input_fn=_local_tensors_to_sharded,
        output_fn=parallel_style._prepare_output
    )

    # send buffer to local rank
    attn.bias.data = attn.bias.data.to(dist.get_rank())
    attn.masked_bias.data = attn.masked_bias.data.to(dist.get_rank())

    return attn



def _parallelize_gpt2_mlp(module, device_mesh): # TODO:
    """
        Reproduction of torch.distributed.tensor.parallel._parallelize_mlp
        on GPT2MLP that is used in transformers' GPT2.
        Parallelize the first layer rowwise, then columnwise.

        There are NO buffers in GPT2MLP.

        Input can be any DTensor, output is local tensor.

        Arguments:
            module: GPT2MLP,   TODO: should add more cases in the future, so it can be 
                            reused for other models
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized GPT2MLP
    """

    assert (isinstance(module, transformers_modeling_gpt2.GPT2MLP) or
            isinstance(module, transformers_modeling_gptj.GPTJMLP)
    ), "Must be a GPT2MLP instance"

    parallel_style = PairwiseParallel()

    try:
        linear_submodules = list(
            filter(lambda x: isinstance(x, Conv1D), module.children())
        )
        funcs = [_colwise_parallelize_conv1d, _rowwise_parallelize_conv1d]
        assert(len(linear_submodules) == 2)
    except:
        linear_submodules = list(
            filter(lambda x: isinstance(x, nn.Linear), module.children())
        )
        funcs = [_colwise_parallelize_linear, _rowwise_parallelize_linear]
        assert(len(linear_submodules) == 2)

    distribute_module(
        linear_submodules[0],
        device_mesh,
        funcs[0],
        input_fn=parallel_style._prepare_input
    )
    distribute_module(
        linear_submodules[1],
        device_mesh,
        funcs[1],
        output_fn=parallel_style._prepare_output
    )
    return module

# GPT-J

def _parallelize_gptj_attn(attn, device_mesh): # TODO:
    """
        Parallelization of GPTJAttention layer. The layer has 4 submodules, and 
        2 buffers: k_/q_/v_proj, out_proj; masked_bias, bias.

        For the buffers, we use a local copy of both buffers on each gpu.
        
        We shard everything head-wise. No need to rearrange, since the key-query-values are split.

        Input can be any DTensor, output is local tensor.

        Arguments:
            module: GPTJAttention,
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized GPTJAttention
    """

    # Check if heads can be split evenly over the GPUs
    world_size = dist.get_world_size()
    assert attn.num_attention_heads % world_size == 0, (
        f"Number of heads ({attn.num_attention_heads}) must be divisible by world size({dist.get_world_size()})"
    )

    attn.num_attention_heads = attn.num_attention_heads // world_size

    parallel_style = PairwiseParallel()
    distribute_module(
        attn.k_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.q_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.v_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.out_proj,
        device_mesh,
        _rowwise_parallelize_linear,
        input_fn=_local_tensors_to_sharded,
        output_fn=parallel_style._prepare_output
    )

    # send buffer to local rank
    attn.bias.data = attn.bias.data.to(dist.get_rank())
    attn.masked_bias.data = attn.masked_bias.data.to(dist.get_rank())

    return attn

# MISTRAL 

def _parallelize_mistral_attn(attn, device_mesh): # TODO:
    """
        Parallelization of MistralAttention layer. The layer has 4 submodules, and 
        2 buffers: k_/q_/v_proj, out_proj; masked_bias, bias.

        Works with LlamaAttention too!

        For the buffers, we use a local copy of both buffers on each gpu.
        
        We shard everything head-wise. No need to rearrange, since the key-query-values are split.

        Input can be any DTensor, output is local tensor.

        Arguments:
            module: Union[MistralAttention, LlamaAttention],
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized attention layer
    """
    assert (
        isinstance(attn, transformers_modeling_mistral.MistralAttention) or
        isinstance(attn, transformers_modeling_llama.LlamaAttention)
    ), "Must be a MistralAttention or LlamaAttention instance"

    # Check if heads can be split evenly over the GPUs
    world_size = dist.get_world_size()
    assert attn.num_key_value_heads % world_size == 0, (
        f"Number of heads ({attn.num_key_value_heads}) must be divisible by world size ({world_size})"
    )

    attn.num_heads = attn.num_heads // world_size
    attn.num_key_value_heads = attn.num_key_value_heads // world_size
    attn.hidden_size = attn.hidden_size // world_size # due to a stupid feature in the code we have to do this
    # num_key_value_groups remains the same

    # avoid dropout
    attn.training = False
    attn.attention_dropout = 0.0

    def _rearrange(mat, n_groups, world_size, dim):
        out_dim, _ = mat.size()
        assert(out_dim % (n_groups * world_size) == 0)

        # split vertically
        result = torch.split(mat, out_dim // (n_groups * world_size), dim=dim)
        idx = []
        for i in range(world_size):
            idx += [i + world_size * j for j in range(n_groups)]

        # rearrange
        result = [result[j] for j in idx]

        # concatenate and return
        return torch.cat(result, dim=dim)

    parallel_style = PairwiseParallel()
    distribute_module(
        attn.k_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    #attn.q_proj.weight.data = _rearrange(attn.q_proj.weight.data, attn.num_key_value_groups, world_size, 0)
    distribute_module(
        attn.q_proj,  # and rearrange here
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.v_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    #attn.o_proj.weight.data = _rearrange(attn.o_proj.weight.data, attn.num_key_value_groups, world_size, 1)
    distribute_module(
        attn.o_proj, # rearrange here
        device_mesh,
        _rowwise_parallelize_linear,
        input_fn=_local_tensors_to_sharded,
        output_fn=parallel_style._prepare_output
    )

    # send rotary embedding buffer to local rank
    attn.rotary_emb.inv_freq.data = attn.rotary_emb.inv_freq.data.to(dist.get_rank())

    if isinstance(attn, transformers_modeling_mistral.MistralAttention):
        attn.rotary_emb.cos_cached.data = attn.rotary_emb.cos_cached.data.to(dist.get_rank())
        attn.rotary_emb.sin_cached.data = attn.rotary_emb.sin_cached.data.to(dist.get_rank())
    elif isinstance(attn, transformers_modeling_llama.LlamaAttention):
        attn.rotary_emb._cos_cached.data = attn.rotary_emb._cos_cached.data.to(dist.get_rank())
        attn.rotary_emb._sin_cached.data = attn.rotary_emb._sin_cached.data.to(dist.get_rank())

    return attn


def _parallelize_mistral_mlp(module, device_mesh): # TODO:
    """
        Reproduction of torch.distributed.tensor.parallel._parallelize_mlp
        on GPT2MLP that is used in transformers' GPT2.
        Parallelize the first layer rowwise, then columnwise.

        There are NO buffers in GPT2MLP.

        Input can be any DTensor, output is local tensor.

        Arguments:
            module: MistralMLP,   TODO: should add more cases in the future, so it can be 
                            reused for other models
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized MistralMLP
    """

    assert (
        isinstance(module, transformers_modeling_mistral.MistralMLP) or
        isinstance(module, transformers_modeling_llama.LlamaMLP)
    ), "Must be a MistralMLP or LlamaMLP instance"

    parallel_style = PairwiseParallel()

    distribute_module(
        module.gate_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )
    distribute_module(
        module.up_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )
    distribute_module(
        module.down_proj,
        device_mesh,
        _rowwise_parallelize_linear,
        input_fn=_local_tensors_to_sharded,
        output_fn=parallel_style._prepare_output
    )
    return module

# OPT

def _parallelize_opt_attn(attn, device_mesh): # TODO:
    """
        Parallelization of OPTAttention layer. Works the same way as for GPT-J.

        Arguments:
            module: OPTAttention,
            device_mesh: DeviceMesh,    must be 1d.
        Output:
            parallelized OPTAttention
    """
    assert (
        isinstance(attn, transformers_modeling_opt.OPTAttention)
    ), "Must be a OPTAttention instance"

    # Check if heads can be split evenly over the GPUs
    world_size = dist.get_world_size()
    assert attn.num_heads % world_size == 0, (
        f"Number of heads ({attn.num_key_value_heads}) must be divisible by world size ({world_size})"
    )

    attn.num_heads = attn.num_heads // world_size
    attn.embed_dim = attn.embed_dim // world_size

    parallel_style = PairwiseParallel()
    distribute_module(
        attn.k_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.q_proj,  # and rearrange here
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.v_proj,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input,
        output_fn=_to_local_tensor
    )

    distribute_module(
        attn.out_proj, # rearrange here
        device_mesh,
        _rowwise_parallelize_linear,
        input_fn=_local_tensors_to_sharded,
        output_fn=parallel_style._prepare_output
    )

    return attn


def _parallelize_opt_layer(layer, device_mesh):
    assert (
        isinstance(layer, transformers_modeling_opt.OPTDecoderLayer)
    ), "Must be an instance of OPTDecoderLayer"

    # parallelize the attention
    layer.self_attn = _parallelize_opt_attn(layer.self_attn, device_mesh)

    # distribute linear layers fc1 and fc2 as in all MLPs
    parallel_style = PairwiseParallel()

    distribute_module(
        layer.fc1,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input
    )
    distribute_module(
        layer.fc2,
        device_mesh,
        _rowwise_parallelize_linear,
        output_fn=parallel_style._prepare_output
    )

    layer.self_attn_layer_norm = layer.self_attn_layer_norm.to(dist.get_rank())
    layer.final_layer_norm = layer.final_layer_norm.to(dist.get_rank())

    return layer

# ALL

def _parallelize_mlp_attn_replicate_the_rest_recursive(module, device_mesh):
    if (isinstance(module, transformers_modeling_gpt2.GPT2MLP)):

        return _parallelize_gpt2_mlp(module, device_mesh)

    elif (isinstance(module, transformers_modeling_gpt2.GPT2Attention)):

        return _parallelize_gpt2_attn(module, device_mesh)

    elif (isinstance(module, transformers_modeling_gptj.GPTJMLP)):

        return _parallelize_gpt2_mlp(module, device_mesh)

    elif (isinstance(module, transformers_modeling_gptj.GPTJAttention)):

        return _parallelize_gptj_attn(module, device_mesh)

    elif (isinstance(module, transformers_modeling_mistral.MistralMLP)):

        return _parallelize_mistral_mlp(module, device_mesh)

    elif (isinstance(module, transformers_modeling_mistral.MistralAttention)):

        return _parallelize_mistral_attn(module, device_mesh)

    elif (isinstance(module, transformers_modeling_llama.LlamaMLP)):

        return _parallelize_mistral_mlp(module, device_mesh)

    elif (isinstance(module, transformers_modeling_llama.LlamaAttention)):

        return _parallelize_mistral_attn(module, device_mesh)
    
    elif (isinstance(module, transformers_modeling_opt.OPTDecoderLayer)):

        return _parallelize_opt_layer(module, device_mesh)

    elif (isinstance(module, nn.LayerNorm) or
          isinstance(module, nn.Dropout) or
          isinstance(module, nn.Embedding) or
          isinstance(module, Conv1D) or 
          isinstance(module, nn.Linear) or
          isinstance(module, transformers_modeling_mistral.MistralRMSNorm) or
          isinstance(module, transformers_modeling_llama.LlamaRMSNorm) or
          "activation" in type(module).__name__.lower()):

            return module.to(dist.get_rank()) # just use local model
    else:

        for n, m in module.named_children():
            module.register_module(
                n, _parallelize_mlp_attn_replicate_the_rest_recursive(m, device_mesh)
            )
        return module


def parallelize_language_model(model, device_mesh):
    model = _parallelize_mlp_attn_replicate_the_rest_recursive(model, device_mesh)
    return model


class _IdModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def _get_replicating_module(device_mesh):
    module = _IdModule()
    parallel_style = PairwiseParallel()
    distribute_module(
        module,
        device_mesh,
        _colwise_parallelize_linear,
        input_fn=parallel_style._prepare_input
    )
    return module
