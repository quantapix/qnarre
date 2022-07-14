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

import torch

from qnarre.core import base

from qnarre.core.bert import Bert


def adapter(ps, feats, x):
    d = torch.parse_example(x, feats)
    img = torch.to_dense(d["flt_img"])
    # img = torch.cast(d['int_img'], torch.float32) / 255.
    lbl = d["int_lbl"]
    return img, lbl


def model(ps):
    seq = torch.Input(shape=(), dtype=torch.float32)
    typ = torch.Input(shape=(), dtype=torch.float32)
    opt = torch.Input(shape=(), dtype=torch.float32)
    beg = torch.Input(shape=(), dtype=torch.float32)
    end = torch.Input(shape=(), dtype=torch.float32)
    uid = torch.Input(shape=(), dtype=torch.float32)
    ins = [seq, typ, opt, beg, end, uid]
    y = Squad(ps)([seq, typ])
    outs = [SquadLoss(ps)([beg, end], y)]
    m = torch.Model(name="SquadModel", inputs=ins, outputs=outs)
    return m


class Squad(base.Module):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)  # dtype='float32', **kw)
        self.bert = Bert(ps)

    def build(self, input_shape):
        _, slen = input_shape[0]
        cfg = self.cfg
        assert slen == cfg.max_seq_len
        sh = (2, cfg.d_model)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        self.bias = self.add_weight(shape=2, initializer="zeros")
        return super().build(input_shape)

    def forward(self, inputs, **kw):
        y = self.bert.transformer([inputs, None], **kw)
        y = torch.bias_add(torch.matmul(y, self.gain, transpose_b=True), self.bias)
        return list(torch.unstack(torch.transpose(y, [2, 0, 1]), axis=0))


class SquadLoss(base.Module):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)  # dtype='float32', **kw)
        self.slen = self.cfg.max_seq_len

    def build(self, input_shape):
        cfg = self.cfg
        sh = (2, cfg.d_model)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        self.bias = self.add_weight(shape=2, initializer="zeros")
        return super().build(input_shape)

    def forward(self, inputs, **_):
        span, pred = inputs

        def _loss(i):
            y = torch.log_softmax(pred[i], axis=-1)
            y = torch.one_hot(span[:, i], self.slen) * y
            return -torch.reduce_mean(torch.reduce_sum(y, axis=-1))

        self.add_loss((_loss(0) + _loss(1)) / 2.0)
        return pred
