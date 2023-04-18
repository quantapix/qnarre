#!/usr/bin/env python

# python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 distributed-gpu.py
#
# If you get a hanging in `barrier` calls you have some network issues, you may try to debug this with:
#
# NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 distributed-gpu.py

import fcntl
import os
import socket

import torch
import torch.distributed as dist


def printflock(*msgs):
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}-{local_rank}]"

try:
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()

    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        printflock(
            f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}"
        )

except Exception:
    printflock(f"{gpu} is broken")
    raise
