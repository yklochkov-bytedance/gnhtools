import torch

import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh

import os
from tqdm import tqdm

# Random utils for handling mutltiprocessing and device meshes

# starting and clearing

def setup(rank, world_size):
    pass

def cleanup():
    pass

def make_device_mesh(world_size):
    return None

# printing with rank

def print_rank(*s):
    """
        Prints with indication of source (rank)
    """
    print(f"[cuda]:", *s)

def print_memory(rank):
    """
        Prints GPU memory allocated for each rank in MB
    """
    mem = torch.cuda.max_memory_allocated(device=f"cuda:{rank}")
    print(f"cuda: memory allocated: {mem / 1024 / 1024 :.1f}MB")

def share_params_and_buffers(model):
    """
        not required
    """
    pass

def get_rank():
    # all on one gpu
    #
    return 0

# broadcasting dataloader

class BroadcastingDataLoader:
    """
        This dataloader wrapper allows to load data in a `src` process and
        broadcast the batch to the rest of the processes.

        Use it to avoid different processes reading the same memory at
        the same time. In addition, in calling `torch.utils.data.DataLoader` 
        can create its own processes for efficient data loading, and creating
        a copy in each process may lead to conflict, e.g. using the same port.
    """
    def __init__(self, dataset, src, *args, use_tqdm=False, **kwargs):
        assert(src == 0)
        self.use_tqdm = use_tqdm
        self.loader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
        self.length = len(self.loader)

    def __len__(self):
        return self.length

    def __iter__(self):
        loader = self.loader
        if self.use_tqdm:
            self._loader = tqdm(self.loader)
        else:
            self._loader = self.loader
        for batch in self._loader:
            # now yield it
            yield batch

    def set_postfix(self, **kwargs):
        """
            Warning: does not do anything if use_tqdm=False
        """
        if self.use_tqdm:
            self._loader.set_postfix(**kwargs)
