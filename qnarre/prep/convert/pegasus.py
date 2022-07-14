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

import os
import tensorflow as tf
import torch

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from transformers import PegasusTokenizer
from transformers.models.pegasus.configuration_pegasus import DEFAULTS, task_params

from ..config.pegasus import PreTrained
from ...run.pegasus import ForConditionalGen


PATTERNS = [
    ["memory_attention", "encoder_attn"],
    ["attention", "attn"],
    ["/", "."],
    [".LayerNorm.gamma", "_layer_norm.weight"],
    [".LayerNorm.beta", "_layer_norm.bias"],
    ["r.layer_", "r.layers."],
    ["output_proj", "out_proj"],
    ["ffn.dense_1.", "fc2."],
    ["ffn.dense.", "fc1."],
    ["ffn_layer_norm", "final_layer_norm"],
    ["kernel", "weight"],
    ["encoder_layer_norm.", "encoder.layer_norm."],
    ["decoder_layer_norm.", "decoder.layer_norm."],
    ["embeddings.weights", "shared.weight"],
]


def rename_state_dict_key(k):
    for n, t in PATTERNS:
        k = k.replace(n, t)
    return k


def convert_pegasus(tf_weights, cfg_updates):
    cfg_kw = DEFAULTS.copy()
    cfg_kw.update(cfg_updates)
    cfg = PreTrained(**cfg_kw)
    m = ForConditionalGen(cfg)
    sd = m.model.state_dict()
    mapping = {}
    for k, v in tf_weights.items():
        new_k = rename_state_dict_key(k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if "dense" in k or "proj" in new_k:
            v = v.T
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"
    mapping["shared.weight"][cfg.PAD] = torch.zeros_like(mapping["shared.weight"][cfg.PAD + 1])
    mapping["encoder.embed_tokens.weight"] = mapping["shared.weight"]
    mapping["decoder.embed_tokens.weight"] = mapping["shared.weight"]
    empty_biases = {
        k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping
    }
    mapping.update(**empty_biases)
    missing, extra = m.model.load_state_dict(mapping, strict=False)
    unexpected_missing = [
        k
        for k in missing
        if k not in ["encoder.embed_positions.weight", "decoder.embed_positions.weight"]
    ]
    assert unexpected_missing == []
    assert extra == []
    return m


def get_tf_weights_as_numpy(path="./ckpt/aeslc/model.ckpt-32000"):
    xs = tf.train.list_variables(path)
    ys = {}
    ignore_name = ["Adafactor", "global_step"]
    for n, shape in tqdm(xs, desc="converting tf checkpoint to dict"):
        if any([x in n for x in ignore_name]):
            continue
        ys[n] = tf.train.load_variable(path, n)
    return ys


def to_pytorch(ckpt_path, save_path):
    dataset = Path(ckpt_path).parent.name
    desired_max_model_length = task_params[f"sum_{dataset}"]["n_pos"]
    tok = PegasusTokenizer.from_pretrained(
        "sshleifer/pegasus", model_max_length=desired_max_model_length
    )
    assert tok.model_max_length == desired_max_model_length
    tok.save_pretrained(save_path)
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    cfg_updates = task_params[f"sum_{dataset}"]
    if dataset == "large":
        cfg_updates["task_params"] = task_params
    torch_model = convert_pegasus(tf_weights, cfg_updates)
    torch_model.save_pretrained(save_path)
    sd = torch_model.state_dict()
    sd.pop("model.decoder.embed_positions.weight")
    sd.pop("model.encoder.embed_positions.weight")
    torch.save(sd, Path(save_path) / "pytorch_model.bin")


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("src_path", type=str)
    x.add_argument("save_path", default=None, type=str)
    y = x.parse_args()
    if y.save_path is None:
        y.save_path = os.path.join("pegasus", Path(y.src_path).parent.name)
    to_pytorch(y.src_path, y.save_path)
