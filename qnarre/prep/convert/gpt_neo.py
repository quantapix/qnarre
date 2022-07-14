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

import json
import re
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath
from transformers.utils import logging

from ..config.gpt_neo import PreTrained
from ...models.gpt_neo import ForCausal


logging.set_verbosity_info()

log = logging.get_logger(__name__)


def load_src_weights(model, config, gpt_neo_checkpoint_path):
    tf_path = abspath(gpt_neo_checkpoint_path)
    log.info(f"Converting TensorFlow checkpoint from {tf_path}")
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if "global_step" not in name and "adam" not in name:
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            name = name.replace("attn/q", "attn/attention/q_proj/w")
            name = name.replace("attn/k", "attn/attention/k_proj/w")
            name = name.replace("attn/v", "attn/attention/v_proj/w")
            name = name.replace("attn/o", "attn/attention/out_proj/w")
            name = name.replace("norm_1", "ln_1")
            name = name.replace("norm_2", "ln_2")
            name = name.replace("attn/compute_output_bias/o_b", "attn/attention/out_proj/b")
            name = name.replace("conv1d_main/c_fc/kernel", "c_fc/w")
            name = name.replace("conv1d_main/c_fc/bias", "c_fc/b")
            name = name.replace("conv1d_main/c_proj/kernel", "c_proj/w")
            name = name.replace("conv1d_main/c_proj/bias", "c_proj/b")
            names.append(name)
            arrays.append(array)
    for name, array in zip(names, arrays):
        name = name[5:]  # skip "gpt2/"
        name = name.split("/")
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if name[-1] == "w" and name[-2] in [
            "out_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "c_proj",
            "c_fc",
        ]:
            array = array.transpose()
        if name == ["wte"]:
            array = array[: config.s_vocab]
        if pointer.shape != array.shape:
            raise ValueError(
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}"
            )
        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model


def to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    config_json = json.load(open(config_file, "r"))
    cfg = GPTNeoConfig(
        d_hidden=config_json["n_embd"],
        n_lays=config_json["n_lays"],
        n_heads=config_json["n_heads"],
        attention_types=config_json["attention_types"],
        n_pos=config_json["n_pos"],
        drop_resid=config_json["res_dropout"],
        drop_embed=config_json["drop_embed"],
        drop_attn=config_json["attn_dropout"],
    )
    print(f"Building from config: {cfg}")
    m = ForCausal(cfg)
    load_src_weights(m, cfg, tf_checkpoint_path)
    print(f"Saving to: {pytorch_dump_path}")
    m.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path)
