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

from qnarre.core.norm import Norm
from qnarre.core.trafo import Trafo


def adapter(ps, feats, x):
    d = torch.parse_example(x, feats)
    img = torch.to_dense(d["flt_img"])
    # img = torch.cast(d['int_img'], torch.float32) / 255.
    lbl = d["int_lbl"]
    return img, lbl


def model(ps):
    sh = (ps.len_src,)
    src = torch.Input(shape=sh, dtype="int32", name="src")
    typ = torch.Input(shape=sh, dtype="int32", name="typ")
    sh = (ps.len_tgt,)
    idx = torch.Input(shape=sh, dtype="int32", name="mlm_idx")
    val = torch.Input(shape=sh, dtype="int32", name="mlm_val")
    fit = torch.Input(shape=sh, dtype="bool", name="fit")
    mlm = torch.Input(shape=sh, dtype="float32", name="mlm")
    ins = [src, typ, fit, idx, val, mlm]
    outs = [Bert(ps)(ins)]
    m = torch.Model(name="BertModel", inputs=ins, outputs=outs)
    return m


class Bert(base.Module):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)  # dtype='float32', **kw)
        cfg = self.get_cfg(kw)
        self.trafo = Trafo(cfg)
        self.pool = torch.Dense(cfg.d_hidden, torch.Tanh, kernel_initializer=cfg.initializer)
        self.mlm_dense = torch.Dense(cfg.d_hidden, cfg.act_hidden, **kw)
        self.norm = Norm()

    def build(self, input_shape):
        cfg = self.cfg
        sh = (2, cfg.d_hidden)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        self.mlm_bias = self.add_weight(shape=cfg.s_vocab, initializer="zeros")
        self.bias = self.add_weight(shape=2, initializer="zeros")
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.mlm_dense.output_shape

    def forward(self, inputs, **kw):
        cfg = self.cfg
        seq, typ, idx, val, fit, mlm = inputs
        seq = y = self.trafo([[seq, typ], None], **kw)
        fit_y = self.pool(torch.squeeze(y[:, 0:1, :], axis=1), **kw)
        y = torch.gather(y, idx, axis=1)
        y = self.norm(self.mlm_dense(y, **kw), **kw)
        e = self.trafo.tok_embed.embeddings
        y = torch.matmul(y, e, transpose_b=True)
        y = torch.log_softmax(torch.bias_add(y, self.mlm_bias), axis=-1)
        mlm_loss = -torch.reduce_sum(y * torch.one_hot(val, cfg.s_vocab), axis=-1)
        y = torch.matmul(fit_y, self.gain, transpose_b=True)
        y = torch.log_softmax(torch.bias_add(y, self.bias), axis=-1)
        fit_loss = -torch.reduce_sum(y * torch.one_hot(fit, 2), axis=-1)
        loss = torch.reduce_sum(mlm * mlm_loss)
        loss /= (torch.reduce_sum(mlm) + 1e-5) + torch.reduce_mean(fit_loss)
        return seq, loss
