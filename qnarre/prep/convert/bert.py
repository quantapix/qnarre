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
import re
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath
from transformers.utils import logging

from ..config.bert import PreTrained
from ...models.bert import ForPreTraining

logging.set_verbosity_info()

log = logging.get_logger(__name__)

_SKIP = [
    "adam_v",
    "adam_m",
    "AdamWeightDecayOptimizer",
    "AdamWeightDecayOptimizer_1",
    "global_step",
]


def load_src_weights(model, src_path):
    src_path = abspath(src_path)
    log.info(f"Loading from: {src_path}")
    xs = tf.train.list_variables(src_path)
    assert len(xs) > 0
    ns, ws = _load_weights(xs, src_path)
    for n, w in zip(ns, ws):
        ss = n.split("/")
        if any(x in _SKIP for x in ss):
            log.info(f"Skipping {'/'.join(ss)}")
            continue
        p = model
        for s in ss:
            if re.fullmatch(r"[A-Za-z]+_\d+", s):
                scopes = re.split(r"_(\d+)", s)
            else:
                scopes = [s]
            if scopes[0] == "kernel" or scopes[0] == "gamma":
                p = getattr(p, "weight")
            elif scopes[0] == "output_bias" or scopes[0] == "beta":
                p = getattr(p, "bias")
            elif scopes[0] == "output_weights":
                p = getattr(p, "weight")
            elif scopes[0] == "squad":
                p = getattr(p, "classifier")
            else:
                try:
                    p = getattr(p, scopes[0])
                except AttributeError:
                    log.info(f"Skipping {'/'.join(ss)}")
                    continue
            if len(scopes) >= 2:
                p = p[int(scopes[1])]
        if s[-11:] == "_embeddings":
            p = getattr(p, "weight")
        elif s == "kernel":
            w = np.transpose(w)
        assert p.shape == w.shape
        p.data = torch.from_numpy(w)
    return model


def _load_weights(xs, src_path):
    ns = []
    ws = {}
    for n, shape in xs:
        log.info(f"Loading TF weight {n} with shape {shape}")
        ns.append(n)
        ws[n] = tf.train.load_variable(src_path, n)
    return ns, ws


def to_pytorch(src_path, cfg_path, save_path):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    m = ForPreTraining(cfg)
    load_src_weights(m, src_path)
    print(f"Saving to: {save_path}")
    torch.save(m.state_dict(), save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path)
