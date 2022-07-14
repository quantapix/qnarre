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

import math
import numpy as np
import re
import tensorflow as tf
import torch

from os.path import abspath
from argparse import ArgumentParser
from transformers.utils import logging

from ..config.big_bird import PreTrained
from ...models.big_bird import ForPreTraining, ForQA

logging.set_verbosity_info()

log = logging.get_logger(__name__)

_SKIP = [
    "adam_v",
    "adam_m",
    "AdamWeightDecayOptimizer",
    "AdamWeightDecayOptimizer_1",
    "global_step",
]


def load_src_weights(model, src_path, is_trivia=False):
    src_path = abspath(src_path)
    log.info(f"Loading from: {src_path}")
    xs = tf.saved_model.load(src_path).variables if is_trivia else tf.train.list_variables(src_path)
    assert len(xs) > 0
    pt_names = list(model.state_dict().keys())
    if is_trivia:
        ns, ws = _load_trivia(xs)
    else:
        ns, ws = _load_weights(xs, src_path)
    for n in ns:
        xs = n.split("/")
        if any(x in _SKIP for x in xs):
            log.info(f"Skipping {'/'.join(xs)}")
            continue
        ts = []
        p = model
        for x in xs:
            if re.fullmatch(r"[A-Za-z]+_\d+", x):
                scopes = re.split(r"_(\d+)", x)
            else:
                scopes = [x]
            if scopes[0] == "kernel" or scopes[0] == "gamma":
                p = getattr(p, "weight")
                ts.append("weight")
            elif scopes[0] == "output_bias" or scopes[0] == "beta":
                p = getattr(p, "bias")
                ts.append("bias")
            elif scopes[0] == "output_weights":
                p = getattr(p, "weight")
                ts.append("weight")
            elif scopes[0] == "squad":
                p = getattr(p, "classifier")
                ts.append("classifier")
            elif scopes[0] == "transform":
                p = getattr(p, "transform")
                ts.append("transform")
                if ("bias" in xs) or ("kernel" in xs):
                    p = getattr(p, "dense")
                    ts.append("dense")
                elif ("beta" in xs) or ("gamma" in xs):
                    p = getattr(p, "LayerNorm")
                    ts.append("LayerNorm")
            else:
                try:
                    p = getattr(p, scopes[0])
                    ts.append(f"{scopes[0]}")
                except AttributeError:
                    log.info(f"Skipping {x}")
                    continue
            if len(scopes) >= 2:
                i = int(scopes[1])
                p = p[i]
                ts.append(f"{i}")
        w = ws[n]
        if x[-11:] == "_embeddings" or x == "embeddings":
            p = getattr(p, "weight")
            ts.append("weight")
        elif x == "kernel":
            w = np.transpose(w)
        if len(w.shape) > len(p.shape) and math.prod(w.shape) == math.prod(p.shape):
            if (
                n.endswith("attention/self/key/kernel")
                or n.endswith("attention/self/query/kernel")
                or n.endswith("attention/self/value/kernel")
            ):
                w = w.transpose(1, 0, 2).reshape(p.shape)
            elif n.endswith("attention/output/dense/kernel"):
                w = w.transpose(0, 2, 1).reshape(p.shape)
            else:
                w = w.reshape(p.shape)
        assert p.shape == w.shape
        t = ".".join(ts)
        log.info(f"Initialize {t} from {n}")
        p.data = torch.from_numpy(w)
        ws.pop(n, None)
        pt_names.remove(t)
    log.info(f"Not copied: {', '.join(ws.keys())}.")
    log.info(f"Not initialized: {', '.join(pt_names)}.")
    return model


def _load_weights(xs, src_path):
    ns = []
    ws = {}
    for n, shape in xs:
        n = n.replace("bert/encoder/LayerNorm", "bert/embeddings/LayerNorm")
        log.info(f"Loading TF weight {n} with shape {shape}")
        ns.append(n)
        ws[n] = tf.train.load_variable(src_path, n)
    return ns, ws


_MAP = {
    "big_bird_attention": "attention/self",
    "output_layer_norm": "output/LayerNorm",
    "attention_output": "attention/output/dense",
    "output": "output/dense",
    "self_attention_layer_norm": "attention/output/LayerNorm",
    "intermediate": "intermediate/dense",
    "tok_embed": "bert/embeddings/tok_embed",
    "pos_embed": "bert/embeddings/pos_embed",
    "type_embeddings": "bert/embeddings/token_type_embeddings",
    "embeddings": "bert/embeddings",
    "layer_normalization": "output/LayerNorm",
    "layer_norm": "LayerNorm",
    "trivia_qa_head": "qa_classifier",
    "dense": "intermediate/dense",
    "dense_1": "qa_outputs",
}


def _load_trivia(xs):
    ns = []
    ws = {}
    for i, x in enumerate(xs):
        ks = x.name.split("/")
        if "transformer_scaffold" in ks[0]:
            ls = ks[0].split("_")
            if len(ls) < 3:
                ls += [0]
            ks[0] = f"bert/encoder/layer_{ls[2]}"
        n = "/".join([_MAP[k] if k in _MAP else k for k in ks])[:-2]
        if "self/attention/output" in n:
            n = n.replace("self/attention/output", "output")
        if i >= len(xs) - 2:
            n = n.replace("intermediate", "output")
        log.info(f"Loading TF weight {n} with shape {x.shape}")
        ns.append(n)
        ws[n] = x.value().numpy()
    return ns, ws


def to_pytorch(src_path, cfg_path, save_path, is_trivia):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    if is_trivia:
        m = ForQA(cfg)
    else:
        m = ForPreTraining(cfg)
    load_src_weights(m, src_path, is_trivia=is_trivia)
    print(f"Saving to: {save_path}")
    m.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--is_trivia", action="store_true")
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path, y.is_trivia)
