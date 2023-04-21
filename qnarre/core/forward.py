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

from torch import nn

from . import output as qo


def forward_masked(self, x, labels=None, **kw):
    yo = self.get_y_opts(**kw)
    ys = self.model(x, **kw, yo=yo)
    y = self.proj(ys[0])
    loss = None
    if labels is not None:
        loss = nn.CrossEntropyLoss()(y.view(-1, self.cfg.s_vocab), labels.view(-1))
    ys = (y,) + ys[1:] + (loss,)
    return qo.WithLoss(*ys) if yo.kw else ys


def forward_qa(self, x, beg=None, end=None, **kw):
    yo = self.get_y_opts(**kw)
    if beg is not None and end is not None:
        yo.cache = False
    ys = self.model(x, **kw, yo=yo)
    b, e = self.proj(ys[0]).split(1, dim=-1)
    b = b.squeeze(-1).contiguous()
    e = e.squeeze(-1).contiguous()
    loss = None
    if beg is not None and end is not None:
        if len(beg.size()) > 1:
            beg = beg.squeeze(-1)
        if len(end.size()) > 1:
            end = end.squeeze(-1)
        i = b.size(1)
        f = nn.CrossEntropyLoss(ignore_index=i)
        beg.clamp_(0, i)
        end.clamp_(0, i)
        loss = (f(b, beg) + f(e, end)) / 2
    ys = (b, e) + ys[1:] + (loss,)
    return qo.LossQA(*ys) if yo.kw else ys
    # return qo.LossSeq2SeqQA(*ys) if yo.kw else ys


def forward_seq(self, x, labels=None, **kw):
    cfg = self.cfg
    yo = self.get_y_opts(**kw)
    if labels is not None:
        yo.cache = False
    ys = self.model(x, **kw, yo=yo)
    y = self.proj(ys[1])  # ys[0][:, 0]; ys[0][:, 0, :]; pooled_output
    loss = None
    if labels is not None:
        if cfg.problem is None:
            dt = labels.dtype
            if cfg.n_labels == 1:
                cfg.problem = "regression"
            elif cfg.n_labels > 1 and (dt == torch.long or dt == torch.int):
                cfg.problem = "single_label"
            else:
                cfg.problem = "multi_label"
        if cfg.problem == "regression":
            if cfg.n_labels == 1:
                loss = nn.MSELoss()(y.squeeze(), labels.squeeze())
            else:
                loss = nn.MSELoss()(y, labels)
        elif cfg.problem == "single_label":
            loss = nn.CrossEntropyLoss()(y.view(-1, cfg.n_labels), labels.view(-1))
        elif cfg.problem == "multi_label":
            loss = nn.BCEWithLogitsLoss()(y, labels)
    ys = (y,) + ys[2:] + (loss,)  # ys[1:]
    return qo.WithLoss(*ys) if yo.kw else ys
    # return qo.LossSeq2Seq(*ys) if yo.kw else ys


def forward_tok(self, x, mask=None, labels=None, **kw):
    cfg = self.cfg
    yo = self.get_y_opts(**kw)
    ys = self.model(x, **kw, yo=yo)
    y = self.proj(ys[0])
    loss = None
    if labels is not None:
        labels = labels.to(y.device)
        f = nn.CrossEntropyLoss()
        l = labels.view(-1)
        if mask is not None:
            l = torch.where(mask.view(-1) == 1, l, torch.tensor(f.ignore_index).type_as(l))
        loss = f(y.view(-1, cfg.n_labels), l)
    ys = (y,) + ys[2:] + (loss,)
    return qo.WithLoss(*ys) if yo.kw else ys
