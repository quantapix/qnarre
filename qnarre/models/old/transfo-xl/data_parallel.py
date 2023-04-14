from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print("obj", obj.size())
                print("dim", dim)
                print("chunk_sizes", chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kw, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kw dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kw = scatter(kw, target_gpus, chunk_sizes, dim) if kw else []
    if len(inputs) < len(kw):
        inputs.extend([() for _ in range(len(kw) - len(inputs))])
    elif len(kw) < len(inputs):
        kw.extend([{} for _ in range(len(inputs) - len(kw))])
    inputs = tuple(inputs)
    kw = tuple(kw)
    return inputs, kw


class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kw):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kw)

    def forward(self, *inputs, **kw):
        if not self.device_ids:
            return self.module(*inputs, **kw)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kw = self.scatter(inputs, kw, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kw[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kw)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kw):
        return parallel_apply(replicas, inputs, kw, device_ids)

    def scatter(self, inputs, kw, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kw, device_ids)
        return scatter_kwargs(inputs, kw, device_ids, chunk_sizes, dim=self.dim)
