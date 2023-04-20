import argparse
import os

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
