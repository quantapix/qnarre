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

from ..config.t5 import PreTrained
from ...models.t5 import ForCondGen


logging.set_verbosity_info()


log = logging.get_logger(__name__)

_SKIP = [
    "adam_v",
    "adam_m",
    "AdamWeightDecayOptimizer",
    "AdamWeightDecayOptimizer_1",
    "global_step",
]


def load_src_weights(model, config, src_path):
    src_path = abspath(src_path)
    log.info(f"Loading from: {src_path}")
    xs = tf.train.list_variables(src_path)
    names = []
    tf_weights = {}
    for name, shape in xs:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(src_path, name)
        names.append(name)
        tf_weights[name] = array
    for txt_name in names:
        name = txt_name.split("/")
        if any(n in _SKIP for n in name):
            log.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            log.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        p = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scopes = re.split(r"_(\d+)", m_name)
            else:
                scopes = [m_name]
            if scopes[0] in ["kernel", "scale", "embedding"]:
                p = getattr(p, "weight")
            elif scopes[0] == "self_attention":
                p = getattr(p, "layer")
                p = p[0]
            elif scopes[0] == "enc_dec_attention":
                p = getattr(p, "layer")
                p = p[1]
            elif scopes[0] == "dense_relu_dense":
                p = getattr(p, "layer")
                p = p[2]
            elif scopes[0] == "rms_norm":
                if hasattr(p, "layer_norm"):
                    p = getattr(p, "layer_norm")
                elif hasattr(p, "final_layer_norm"):
                    p = getattr(p, "final_layer_norm")
            elif scopes[0] == "scale":
                p = getattr(p, "weight")
            elif scopes[0] == "output_bias" or scopes[0] == "beta":
                p = getattr(p, "bias")
            elif scopes[0] == "squad":
                p = getattr(p, "classifier")
            elif scopes[0] == "decoder" and name[1] == "logits":
                continue
            elif scopes[0] == "logits":
                p = getattr(p, "lm_head")
            elif scopes[0] == "wi" and len(scopes) > 1 and scopes[1].isdigit():
                p = getattr(p, f"wi_{scopes[1]}")
                continue
            else:
                try:
                    p = getattr(p, scopes[0])
                except AttributeError:
                    log.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scopes) >= 2:
                p = p[int(scopes[1])]
        if scopes[0] not in ["kernel", "scale", "embedding"]:
            p = getattr(p, "weight")
        if scopes[0] != "embedding":
            log.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        assert p.shape == array.shape
        log.info(f"Initialize PyTorch weight {name}")
        p.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)
    log.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


def to_pytorch(src_path, cfg_path, save_path):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    m = ForCondGen(cfg)
    load_src_weights(m, cfg, src_path)
    print(f"Saving to: {save_path}")
    m.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path)
