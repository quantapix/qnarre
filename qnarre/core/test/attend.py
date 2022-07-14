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
# pytest -s qnarre/neura/layers/attn_test.py

import torch

import qnarre.core.utils as U
import qnarre.core as L

params = dict(
    dim_attn=8,
    dim_attn_k=None,
    dim_attn_v=None,
    dim_hidden=16,
    drop_attn=None,
    drop=0.1,
    n_heads=4,
)


class Owner:
    pre = post = None

    def __init__(self):
        self.ps = U.Params(params).init_comps()
        self.pre = None
        self.post = None
        i = torch.constant([0.0] * (4 * 10), shape=(4, 10))
        self.src_b = torch.Variable(initial_value=i)
        i = torch.constant([0.0] * (4 * 10), shape=(4, 10))
        self.mem_b = torch.Variable(initial_value=i)


def test_owner_none():
    a = L.Attend(Owner())
    a.build([(4, 10, 16)])
    src = torch.constant([0.0] * (4 * 10 * 16), shape=(4, 10, 16))
    a.call([src])
    bias = torch.constant([0.0] * (4 * 10), shape=(4, 10))
    bias = torch.expand_dims(torch.expand_dims(bias, axis=1), axis=3)
    a.call([src, bias])
    ctx = torch.constant([0.0] * (4 * 15 * 16), shape=(4, 15, 16))
    a.call([src, bias, None, ctx])


def test_with_owner():
    a = L.Attend(Owner())
    a.build([(4, 10, 16), (), (4, 18, 16), ()])
    src = torch.constant([0.0] * (4 * 10 * 16), shape=(4, 10, 16))
    bias = torch.constant([0.0] * (4 * 10), shape=(4, 10))
    bias = torch.expand_dims(torch.expand_dims(bias, axis=1), axis=3)
    mem = torch.constant([0.0] * (4 * 15 * 16), shape=(4, 15, 16))
    ctx = torch.constant([0.0] * (4 * 15 * 16), shape=(4, 15, 16))
    a.call([src, bias, mem, ctx])


def test_shift():
    a = L.Attend(Owner())
    x = torch.constant([1, 2, 3, 4, 5, 6], shape=(1, 1, 2, 3))
    torch.print(x)
    x = a.shift(x)
    torch.print(x)
