# Copyright 2022 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import os
import torch

from itertools import chain

from transformers.activations import ACT2FN


def activation(k, v=None):
    return ACT2FN[k] if isinstance(k, str) else (v or k)


def view_2D(*xs):
    return [x.view(-1, x.size(-1)) if x is not None else None for x in xs]


def view_3D(*xs):
    return [x.view(-1, x.size(-2), x.size(-1)) if x is not None else None for x in xs]


def get_list(xs):
    ys = set()
    for x in xs:
        ys = ys | set(x)
    ys = list(ys)
    ys.sort()
    return ys


def group_texts(size, xs):
    ys = {k: list(chain(*xs[k])) for k in xs.keys()}
    n = len(ys[list(xs.keys())[0]])
    if n >= size:
        n = (n // size) * size
    ys = {k: [x[i : i + size] for i in range(0, n, size)] for k, x in ys.items()}
    ys["labels"] = ys["input_ids"].copy()
    return ys


def init_array(xs, dataset, lim):
    i = 0
    ys = np.full((len(dataset), lim), -100, dtype=np.float32)  # float64)
    for x in xs:
        batch = x.shape[0]
        cols = x.shape[1]
        if i + batch < len(dataset):
            ys[i : i + batch, :cols] = x
        else:
            ys[i:, :cols] = x[: len(dataset) - i]
        i += batch
    return ys


def big_neg(dtype=None):
    f = dtype
    return torch.float16.min if f == "float16" else -1e9


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, x):
        if x not in self.word2idx:
            self.idx2word.append(x)
            self.word2idx[x] = len(self.idx2word) - 1
        return self.word2idx[x]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.eval = self.tokenize(os.path.join(path, "eval.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids


def shift_right(x, PAD, dec_START):
    assert PAD is not None
    y = x.new_zeros(x.shape)
    y[:, 1:] = x[:, :-1].clone()
    y[:, 0] = dec_START
    y.masked_fill_(y == -100, PAD)
    return y


def shift_right2(x, PAD):
    assert PAD is not None
    y = x.clone()
    y.masked_fill_(y == -100, PAD)
    eos = (y.ne(PAD).sum(dim=1) - 1).unsqueeze(-1)
    dec_START = y.gather(1, eos).squeeze()
    y[:, 1:] = y[:, :-1].clone()
    y[:, 0] = dec_START
    return y


def causal_mask(shape, dtype, device, c_len=0):  # qpx add device
    b, n = shape
    y = torch.full((n, n), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    cond = torch.arange(y.size(-1), device=device)
    y.masked_fill_(cond < (cond + 1).view(y.size(-1), 1), 0)
    y = y.to(dtype)
    if c_len > 0:
        y = torch.cat([torch.zeros(n, c_len, dtype=dtype, device=device), y], dim=-1)
    return y[None, None, :, :].expand(b, 1, n, n + c_len)


def expand_mask(x, dtype, len=None):
    b, n = x.size()
    len = len if len is not None else n
    y = 1.0 - x[:, None, None, :].expand(b, 1, len, n).to(dtype)
    return y.masked_fill(y.to(torch.bool), torch.finfo(dtype).min)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    expanded_attention_mask = inverted_mask.masked_fill(
        inverted_mask.bool(), torch.finfo(dtype).min
    )
    expanded_attention_mask = expanded_attention_mask * inverted_mask
    return expanded_attention_mask
