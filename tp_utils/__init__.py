"""
Implementation of Megatron style parallelizm for Huggingface models. See https://arxiv.org/pdf/1909.08053.pdf

Parallelizes attention and MLP layers. Does not affect embeddings, dropouts and layer normalizations.

Supported models:
    - GPT2, -medium, -Large, -XL
    - GPT-J
    - Mistral-7B-v0.1

Remark:
This implementation uses Torch v2.0.1+cu117. Some of the classes and functions have to be imported from torch.distributed._tensor,
which in the latest version is depricated. The packages can be imported directly from torch.distributed.__init__.py
in the latest version.
Namely, the classes and functions in question: DeviceMesh, DTensor, distribute_module, distribute_tensor.
"""

__all__ = [
    "parallelize_language_model",
    "setup",
    "cleanup",
    "print_rank",
    "print_memory",
    "make_device_mesh",
    "share_params_and_buffers",
    "get_rank",
    "BroadcastingDataLoader"
]

import torch
if torch.cuda.device_count() > 1:
    print("Distributed regime.")
    from tp_utils.hf_models import parallelize_language_model
    from tp_utils.utils import (
        setup,
        cleanup,
        print_rank,
        print_memory,
        make_device_mesh,
        share_params_and_buffers,
        get_rank,
        BroadcastingDataLoader,
    )
else:
    print("Non-distributed regime.")
    def parallelize_language_model(model, device_mesh):
        assert(device_mesh is None)
        return model.to(0)
    from tp_utils.utils_dummy import (
        setup,
        cleanup,
        print_rank,
        print_memory,
        make_device_mesh,
        share_params_and_buffers,
        get_rank,
        BroadcastingDataLoader,
    )
