import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def run(world_size, rank):
    n = torch.cuda.device_count() // world_size
    ids = list(range(rank * n, (rank + 1) * n))
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {ids} \n",
        end="",
    )
    model = Model().cuda(ids[0])
    ddp = DDP(model, ids)
    loss = nn.MSELoss()
    optimizer = optim.SGD(ddp.parameters(), lr=0.001)

    optimizer.zero_grad()
    ys = ddp(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(ids[0])
    loss(ys, labels).backward()
    optimizer.step()


def run_checkpoint(world_size, rank):
    model = Model().to(rank)
    ddp = DDP(model, device_ids=[rank])
    loss = nn.MSELoss()
    optimizer = optim.SGD(ddp.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp.state_dict(), CHECKPOINT_PATH)
    dist.barrier()
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ddp.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    ys = ddp(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss(ys, labels).backward()
    optimizer.step()

    dist.barrier()
    if rank == 0:
        os.remove(CHECKPOINT_PATH)


class MpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(MpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def run_mp(world_size, rank):
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    model = MpModel(dev0, dev1)
    ddp = DDP(model)
    loss = nn.MSELoss()
    optimizer = optim.SGD(ddp.parameters(), lr=0.001)

    optimizer.zero_grad()
    ys = ddp(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss(ys, labels).backward()
    optimizer.step()


def main(world_size, rank):
    env = {k: os.environ[k] for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
    print(f"[{os.getpid()}] Initializing process group with: {env}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
        end="",
    )
    run(world_size, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local-world-size", type=int, default=1)
    args = parser.parse_args()
    main(args.local_world_size, args.local_rank)
