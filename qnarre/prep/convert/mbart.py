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

from argparse import ArgumentParser
from torch import nn

from ..config.mbart import PreTrained
from ...run.mbart import ForConditionalGen


def make_linear_from_emb(x):
    s_vocab, emb_size = x.weight.shape
    y = nn.Linear(s_vocab, emb_size, bias=False)
    y.weight.data = x.weight.data
    return y


def to_pytorch(src_path, cfg_path, save_path, finetuned=False, mbart_50=False):
    d = torch.load(src_path, map_location="cpu")["model"]
    for k in _IGNORE:
        d.pop(k, None)
    s_vocab = d["encoder.embed_tokens.weight"].shape[0]
    cfg = PreTrained.from_pretrained(cfg_path, s_vocab=s_vocab)
    if mbart_50 and finetuned:
        cfg.act_fun = "relu"
    print(f"Building from config: {cfg}")
    d["shared.weight"] = d["decoder.embed_tokens.weight"]
    m = ForConditionalGen(cfg)
    m.model.load_state_dict(d)
    if finetuned:
        m.lm_head = make_linear_from_emb(m.model.shared)
    print(f"Saving to: {save_path}")
    torch.save(m.state_dict(), save_path)


_IGNORE = [
    "encoder.version",
    "decoder.version",
    "model.encoder.version",
    "model.decoder.version",
    "_float_tensor",
    "decoder.output_projection.weight",
]

if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", type=str)
    x.add_argument("--cfg_path", default="facebook/mbart-large-cc25", type=str)
    x.add_argument("--save_path", default=None, type=str)
    x.add_argument("--finetuned", action="store_true")
    x.add_argument("--mbart_50", action="store_true")
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.finetuned, y.mbart_50)
