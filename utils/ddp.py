from __future__ import annotations
import os
import torch
import torch.distributed as dist


def ddp_is_available() -> bool:
    return dist.is_available()


def ddp_is_initialized() -> bool:
    return dist.is_initialized()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not ddp_is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def barrier():
    if ddp_is_initialized():
        dist.barrier()
