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

import re
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging

from ..config.gpt2 import PreTrained
from ...models.gpt2 import Model

logging.set_verbosity_info()


log = logging.get_logger(__name__)


def load_src_weights(model, src_path):
    src_path = abspath(src_path)
    log.info(f"Loading from: {src_path}")
    xs = tf.train.list_variables(src_path)
    assert len(xs) > 0
    ns, ws = _load_weights(xs, src_path)
    for n, w in zip(ns, ws):
        ss = n[6:].split("/")
        p = model
        for s in ss:
            if re.fullmatch(r"[A-Za-z]+\d+", s):
                scopes = re.split(r"(\d+)", s)
            else:
                scopes = [s]
            if scopes[0] == "w" or scopes[0] == "g":
                p = getattr(p, "weight")
            elif scopes[0] == "b":
                p = getattr(p, "bias")
            elif scopes[0] == "wpe" or scopes[0] == "wte":
                p = getattr(p, scopes[0])
                p = getattr(p, "weight")
            else:
                p = getattr(p, scopes[0])
            if len(scopes) >= 2:
                p = p[int(scopes[1])]
        w = ws[n]
        assert p.shape == w.shape
        p.data = torch.from_numpy(w)
    return model


def _load_weights(xs, src_path):
    ns = []
    ws = {}
    for n, shape in xs:
        log.info(f"Loading TF weight {n} with shape {shape}")
        ns.append(n)
        ws[n] = tf.train.load_variable(src_path, n).squeeze()
    return ns, ws


def to_pytorch(src_path, cfg_path, save_path):
    cfg = PreTrained() if cfg_path == "" else PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    m = Model(cfg)
    load_src_weights(m, src_path)
    w = save_path + "/" + WEIGHTS_NAME
    print(f"Saving to: {w}")
    torch.save(m.state_dict(), w)
    c = save_path + "/" + CONFIG_NAME
    print(f"Saveing config to: {c}")
    with open(c, "w", encoding="utf-8") as f:
        f.write(cfg.to_json_string())


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default="", type=str)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path)
