#!/usr/bin/env python

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import evaluate

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm


def setup(rank, world_size, fn=None, backend="gloo"):  # 'tcp'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if fn is not None:
        fn(rank, world_size)


def cleanup():
    dist.destroy_process_group()


def run_old(rank, size):
    print(dist.get_world_size())
    tensor = torch.ones(1)
    list = [torch.zeros(1) for _ in range(size)]
    # dist.gather(tensor, dst=0, gather_list=list, group=0)
    # print('Rank ', rank, ' has data ', sum(list)[0])


def run(rank, size):
    group = dist.new_group([0, 1, 2, 3])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print("Rank ", rank, " has data ", tensor[0])


def run_blocking(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print("Rank ", rank, " has data ", tensor[0])


def run_nonblocking(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        print("Rank 0 started sending")
    else:
        req = dist.irecv(tensor=tensor, src=0)
        print("Rank 1 started receiving")
    req.wait()
    print("Rank ", rank, " has data ", tensor[0])


def run_model():
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_ds = tokenized["train"].shuffle(seed=42).select(range(1000))
    eval_ds = tokenized["test"].shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(eval_ds, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


def gather(tensor, rank, list=None, root=0, group=None):
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert list is not None
        dist.gather_recv(list, tensor, group)
    else:
        dist.gather_send(tensor, root, group)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=setup, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
