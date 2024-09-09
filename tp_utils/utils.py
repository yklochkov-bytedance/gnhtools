import torch

import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh

import os
from tqdm import tqdm

# Random utils for handling mutltiprocessing and device meshes

# starting and clearing

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"

    # only assign MASTER_PORT if it wasn't already specified externally
    #
    if ("MASTER_PORT" not in os.environ or
        not os.environ["MASTER_PORT"]):
        os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def make_device_mesh(world_size):
    """
        Creates 1D mesh of GPU devices
    """
    return DeviceMesh("cuda", torch.arange(0, world_size))

# printing with rank

def print_rank(*s):
    """
        Prints with indication of source (rank)
    """
    rank = dist.get_rank()
    print(f"[rank {rank}]:", *s)

def print_memory(rank):
    """
        Prints GPU memory allocated for each rank in MB
    """
    mem = torch.cuda.max_memory_allocated(device=f"cuda:{rank}")
    print(f"rank {rank}: memory allocated: {mem / 1024 / 1024 :.1f}MB")
    dist.barrier()

def share_params_and_buffers(model):
    """
        Shares the model data between the processes. Apply before spawning processes,
        and pass the model as argument to the worker.
        See example in tests/test_dist_iter.py

        Input: nn.Module
        Output: None
    """
    for par in model.parameters():
        par.share_memory_()
    for buff in model.buffers():
        buff.share_memory_()

def get_rank():
    # for reproducibility on non-disrtibuted scripts (see utils_dummy.py)
    #
    return dist.get_rank()

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
        self.src = src
        self.use_tqdm = use_tqdm
        if dist.get_rank() == src:
            self.loader = torch.utils.data.DataLoader(dataset, *args, **kwargs)

        if dist.get_rank() == src:
            lst = [len(self.loader)]
        else:
            lst = [None]
        dist.broadcast_object_list(lst, src=self.src)
        self.length = lst[0]

    def __len__(self):
        return self.length

    def __iter__(self):
        if dist.get_rank() == self.src:
            loader = self.loader
            if self.use_tqdm:
                self._loader = tqdm(self.loader)
            else:
                self._loader = self.loader
            for batch in self._loader:
                # broadcast batch
                lst = [batch]
                dist.broadcast_object_list(lst, src=self.src)
                batch = lst[0]
                # now yield it
                yield batch
            lst = [None] # so that we know when to stop
            dist.broadcast_object_list(lst, src=self.src)
        else:
            while True:
                # recieve batch
                lst = [None]
                dist.broadcast_object_list(lst, src=self.src)
                batch = lst[0]
                # check if not None, otherwise end of loop
                if batch is not None:
                    yield batch
                else:
                    break

    def set_postfix(self, **kwargs):
        """
            Warning: does not do anything if use_tqdm=False
        """
        if self.use_tqdm and dist.get_rank() == self.src:
            self._loader.set_postfix(**kwargs)
