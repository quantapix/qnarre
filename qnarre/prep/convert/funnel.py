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
import re
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from os.path import abspath
from transformers.utils import logging

from ..config.funnel import PreTrained
from ...models.funnel import Base, Model


logging.set_verbosity_info()

log = logging.get_logger(__name__)


def load_src_weights(model, config, tf_checkpoint_path):
    tf_path = abspath(tf_checkpoint_path)
    log.info(f"Converting TensorFlow checkpoint from {tf_path}")
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    _layer_map = {
        "k": "k_head",
        "q": "q_head",
        "v": "v_head",
        "o": "post_proj",
        "layer_1": "linear_1",
        "layer_2": "linear_2",
        "rel_attn": "attention",
        "ff": "ffn",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
        "lookup_table": "weight",
        "word_embedding": "word_embeddings",
        "input": "embeddings",
    }
    for name, array in zip(names, arrays):
        name = name.split("/")
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            log.info(f"Skipping {'/'.join(name)}")
            continue
        if name[0] == "generator":
            continue
        pointer = model
        skipped = False
        for m_name in name[1:]:
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(
                r"layer_\d+", m_name
            ):
                layer_index = int(re.search(r"layer_(\d+)", m_name).groups()[0])
                if layer_index < config.n_lays:
                    block_idx = 0
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    layer_index -= config.n_lays
                    pointer = pointer.layers[layer_index]
            elif m_name == "r" and isinstance(pointer, FunnelRelMultiheadAttention):
                pointer = pointer.r_kernel
                break
            elif m_name in _layer_map:
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}", array.shape)
                    skipped = True
                    break
        if not skipped:
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            if m_name == "kernel":
                array = np.transpose(array)
            pointer.data = torch.from_numpy(array)
    return model


def to_pytorch(src_path, cfg_path, save_path, base):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    m = Base(cfg) if base else Model(cfg)
    load_src_weights(m, cfg, src_path)
    print(f"Saving to: {save_path}")
    torch.save(m.state_dict(), save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--base", action="store_true")
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path, y.base)
