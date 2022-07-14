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

from ..config.canine import PreTrained
from ...models.canine import Model

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
                "cls",
                "autoregressive_decoder",
                "char_output_weights",
            ]
            for n in name
        ):
            log.info(f"Skipping {'/'.join(name)}")
            continue
        if name[0] == "bert":
            name[0] = "encoder"
        elif name[1] == "embeddings":
            name.remove(name[1])
        elif name[1] == "segment_embeddings":
            name[1] = "token_type_embeddings"
        elif name[1] == "initial_char_encoder":
            name = ["chars_to_molecules"] + name[-2:]
        elif name[0] == "final_char_encoder" and name[1] in ["LayerNorm", "conv"]:
            name = ["projection"] + name[1:]
        pointer = model
        for m_name in name:
            if (re.fullmatch(r"[A-Za-z]+_\d+", m_name)) and "Embedder" not in m_name:
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    log.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name[-10:] in [f"Embedder_{i}" for i in range(8)]:
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        if pointer.shape != array.shape:
            raise ValueError(
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            )
        log.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def to_pytorch(src_path, save_path):
    cfg = PreTrained()
    m = Model(cfg)
    m.eval()
    print(f"Building from config: {cfg}")
    load_src_weights(m, cfg, src_path)
    print(f"Saving to: {save_path}")
    m.save_pretrained(save_path)
    t = Tokenizer()
    print(f"Save tokenizer files to {save_path}")
    t.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.save_path)
