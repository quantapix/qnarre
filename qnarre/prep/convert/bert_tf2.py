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
from transformers.utils import logging

from ..config.bert import PreTrained
from ...models.bert import Model

logging.set_verbosity_info()

log = logging.get_logger(__name__)


def load_src_weights(model, src_path, config):
    src_path = abspath(src_path)
    log.info(f"Loading from: {src_path}")
    xs = tf.train.list_variables(src_path)
    ns, ws = _load_weights(xs, src_path)
    for n in ns:
        ss = n.split("/")
        p = model
        trace = []
        for s in ss:
            if s == ".ATTRIBUTES":
                break
            if s.startswith("layer_with_weights"):
                layer_num = int(s.split("-")[-1])
                if layer_num <= 2:
                    continue
                elif layer_num == 3:
                    trace.extend(["embeddings", "LayerNorm"])
                    p = getattr(p, "embeddings")
                    p = getattr(p, "LayerNorm")
                elif layer_num > 3 and layer_num < config.n_lays + 4:
                    trace.extend(["encoder", "layer", str(layer_num - 4)])
                    p = getattr(p, "encoder")
                    p = getattr(p, "layer")
                    p = p[layer_num - 4]
                elif layer_num == config.n_lays + 4:
                    trace.extend(["pooler", "dense"])
                    p = getattr(p, "pooler")
                    p = getattr(p, "dense")
            elif s == "embeddings":
                trace.append("embeddings")
                p = getattr(p, "embeddings")
                if layer_num == 0:
                    trace.append("tok_embed")
                    p = getattr(p, "tok_embed")
                elif layer_num == 1:
                    trace.append("pos_embed")
                    p = getattr(p, "pos_embed")
                elif layer_num == 2:
                    trace.append("token_type_embeddings")
                    p = getattr(p, "token_type_embeddings")
                else:
                    raise ValueError("Unknown embedding layer with name {full_name}")
                trace.append("weight")
                p = getattr(p, "weight")
            elif s == "_attention_layer":
                trace.extend(["attention", "self"])
                p = getattr(p, "attention")
                p = getattr(p, "self")
            elif s == "_attention_layer_norm":
                trace.extend(["attention", "output", "LayerNorm"])
                p = getattr(p, "attention")
                p = getattr(p, "output")
                p = getattr(p, "LayerNorm")
            elif s == "_attention_output_dense":
                trace.extend(["attention", "output", "dense"])
                p = getattr(p, "attention")
                p = getattr(p, "output")
                p = getattr(p, "dense")
            elif s == "_output_dense":
                trace.extend(["output", "dense"])
                p = getattr(p, "output")
                p = getattr(p, "dense")
            elif s == "_output_layer_norm":
                trace.extend(["output", "LayerNorm"])
                p = getattr(p, "output")
                p = getattr(p, "LayerNorm")
            elif s == "_key_dense":
                trace.append("key")
                p = getattr(p, "key")
            elif s == "_query_dense":
                trace.append("query")
                p = getattr(p, "query")
            elif s == "_value_dense":
                trace.append("value")
                p = getattr(p, "value")
            elif s == "_intermediate_dense":
                trace.extend(["intermediate", "dense"])
                p = getattr(p, "intermediate")
                p = getattr(p, "dense")
            elif s == "_output_layer_norm":
                trace.append("output")
                p = getattr(p, "output")
            elif s in ["bias", "beta"]:
                trace.append("bias")
                p = getattr(p, "bias")
            elif s in ["kernel", "gamma"]:
                trace.append("weight")
                p = getattr(p, "weight")
            else:
                log.warning(f"Ignored {s}")
        trace = ".".join(trace)
        w = ws[n]
        if re.match(r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace) or re.match(
            r"(\S+)\.attention\.output\.dense\.weight", trace
        ):
            w = w.reshape(p.data.shape)
        if "kernel" in n:
            w = w.transpose()
        assert p.shape == w.shape
        p.data = torch.from_numpy(w)
    return model


def _load_weights(xs, src_path):
    ns = []
    ws = []
    ds = []
    for n, _ in xs:
        ss = n.split("/")
        if n == "_CHECKPOINTABLE_OBJECT_GRAPH" or ss[0] in [
            "global_step",
            "save_counter",
        ]:
            log.info(f"Skipping non-model layer {n}")
            continue
        if "optimizer" in n:
            log.info(f"Skipping optimization layer {n}")
            continue
        if ss[0] == "model":
            ss = ss[1:]
        d = 0
        for s in ss:
            if s.startswith("layer_with_weights"):
                d += 1
            else:
                break
        ds.append(d)
        ns.append("/".join(ss))
        ws[n] = tf.train.load_variable(src_path, n)
    log.info(f"Read {len(ws):,} layers")
    if len(set(ds)) != 1:
        raise ValueError(f"Found layers with different depths (layer depth {list(set(ds))})")
    ds = list(set(ds))[0]
    if ds != 1:
        raise ValueError("Found more than just the embedding/encoder layers")
    return ns, ws


def to_pytorch(src_path, cfg_path, save_path):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    m = Model(cfg)
    load_src_weights(m, src_path, cfg)
    print(f"Saving to: {save_path}")
    torch.save(m.state_dict(), save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", type=str, required=True)
    x.add_argument("--cfg_path", type=str, required=True)
    x.add_argument("--save_path", type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path)
