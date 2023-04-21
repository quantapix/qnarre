import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class Model2(nn.Module):
    def __init__(self, d0, d1):
        super(Model2, self).__init__()
        self.d0 = d0
        self.d1 = d1
        self.net1 = torch.nn.Linear(10, 32).to(d0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(32, 5).to(d1)

    def forward(self, x):
        x = x.to(self.d0)
        x = self.relu(self.net1(x))
        x = x.to(self.d1)
        return self.net2(x)


def run(args):
    r = int(os.environ["LOCAL_RANK"])
    n = torch.cuda.device_count() // int(os.environ["LOCAL_WORLD_SIZE"])
    ids = list(range(r * n, (r + 1) * n))
    print(
        f"[{os.getpid()}] run with: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, n = {n}, device_ids = {ids} \n",
        end="",
    )
    id = ids[0]
    if len(ids) == 1:
        m = DDP(Model().cuda(id), device_ids=ids, output_device=id)
        labels = torch.randn(20, 5).to(id)
    else:
        if args.mesh:
            mesh = DeviceMesh("cuda", ids)
            m = parallelize_module(Model(), mesh, PairwiseParallel())
            labels = torch.randn(20, 5).to(id)
        else:
            m = DDP(Model2(id, ids[1]))
            labels = torch.randn(20, 5).to(ids[1])
    loss = nn.MSELoss()
    o = optim.SGD(m.parameters(), lr=0.001)
    for _ in range(args.iter_nums):
        o.zero_grad()
        ys = m(torch.randn(20, 10).cuda(id))
        loss(ys, labels).backward()
        o.step()


def run_checkpoint(local_world):
    r = int(os.environ["LOCAL_RANK"])
    model = Model().to(r)
    ddp = DDP(model, device_ids=[r])
    loss = nn.MSELoss()
    optimizer = optim.SGD(ddp.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if r == 0:
        torch.save(ddp.state_dict(), CHECKPOINT_PATH)
    dist.barrier()
    map_location = {"cuda:%d" % 0: "cuda:%d" % r}
    ddp.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    ys = ddp(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(r)
    loss(ys, labels).backward()
    optimizer.step()

    dist.barrier()
    if r == 0:
        os.remove(CHECKPOINT_PATH)


def main(args):
    env = {k: os.environ[k] for k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK")}
    print(f"[{os.getpid()}] init_process_group with: {env}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] main with: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
        end="",
    )
    run(args)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iter_nums", type=int, default=2)
    p.add_argument("--mesh", action="store_true")
    args = p.parse_args()
    main(args)

# torchrun --standalone --nproc-per-node=gpu ddp.py

# torchrun --rdzv-id=123 --rdzv-backend=c10d --rdzv-endpoint=localhost:29402 --nnodes=1:2 --nproc-per-node=2 ddp.py
