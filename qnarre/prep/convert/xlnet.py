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
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath, join
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging

from ..config.xlnet import PreTrained
from ...models.xlnet import ForQA, ForSeqClassifier, LMHead


GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


logging.set_verbosity_info()

log = logging.get_logger(__name__)


def build_map(model, cfg, tf_weights=None):
    tf_to_pt_map = {}
    if hasattr(model, "transformer"):
        if hasattr(model, "lm_loss"):
            tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
        if (
            hasattr(model, "sequence_summary")
            and "model/sequnece_summary/summary/kernel" in tf_weights
        ):
            tf_to_pt_map[
                "model/sequnece_summary/summary/kernel"
            ] = model.sequence_summary.summary.weight
            tf_to_pt_map[
                "model/sequnece_summary/summary/bias"
            ] = model.sequence_summary.summary.bias
        if (
            hasattr(model, "logits_proj")
            and cfg.finetune is not None
            and f"model/regression_{cfg.finetune}/logit/kernel" in tf_weights
        ):
            tf_to_pt_map[f"model/regression_{cfg.finetune}/logit/kernel"] = model.logits_proj.weight
            tf_to_pt_map[f"model/regression_{cfg.finetune}/logit/bias"] = model.logits_proj.bias
        model = model.transformer
    tf_to_pt_map.update(
        {
            "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
            "model/transformer/mask_emb/mask_emb": model.mask_emb,
        }
    )
    for i, b in enumerate(model.layer):
        layer_str = f"model/transformer/layer_{i}/"
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.rel_attn.o,
                layer_str + "rel_attn/q/kernel": b.rel_attn.q,
                layer_str + "rel_attn/k/kernel": b.rel_attn.k,
                layer_str + "rel_attn/r/kernel": b.rel_attn.r,
                layer_str + "rel_attn/v/kernel": b.rel_attn.v,
                layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
                layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
                layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
                layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
            }
        )
    if cfg.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update(
        {
            "model/transformer/r_r_bias": r_r_list,
            "model/transformer/r_w_bias": r_w_list,
            "model/transformer/r_s_bias": r_s_list,
            "model/transformer/seg_embed": seg_embed_list,
        }
    )
    return tf_to_pt_map


def load_src_weights(model, config, src_path):
    init_vars = tf.train.list_variables(src_path)
    tf_weights = {}
    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(src_path, name)
        tf_weights[name] = array
    tf_to_pt_map = build_map(model, config, tf_weights)
    for name, p in tf_to_pt_map.items():
        log.info(f"Importing {name}")
        if name not in tf_weights:
            log.info(f"{name} not in tf pre-trained weights, skipping")
            continue
        array = tf_weights[name]
        if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
            log.info("Transposing")
            array = np.transpose(array)
        if isinstance(p, list):
            assert (
                len(p) == array.shape[0]
            ), f"Pointer length {len(p)} and array length {array.shape[0]} mismatched"
            for i, p_i in enumerate(p):
                arr_i = array[i, ...]
                assert p_i.shape == arr_i.shape
                p_i.data = torch.from_numpy(arr_i)
        else:
            assert p.shape == array.shape
            p.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)
    log.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


def to_pytorch(src_path, bert_config_file, save_path, finetune=None):
    cfg = PreTrained.from_json_file(bert_config_file)
    print(f"Building from config: {cfg}")
    finetune = finetune.lower() if finetune is not None else ""
    if finetune in GLUE_TASKS_NUM_LABELS:
        cfg.finetune = finetune
        cfg.n_labels = GLUE_TASKS_NUM_LABELS[finetune]
        m = ForSeqClassifier(cfg)
    elif "squad" in finetune:
        cfg.finetune = finetune
        m = ForQA(cfg)
    else:
        m = LMHead(cfg)
    load_src_weights(m, cfg, src_path)
    w = join(save_path, WEIGHTS_NAME)
    print(f"Saving to: {abspath(w)}")
    torch.save(m.state_dict(), w)
    c = join(save_path, CONFIG_NAME)
    print(f"Saving config to: {abspath(c)}")
    with open(c, "w", encoding="utf-8") as f:
        f.write(cfg.to_json_string())


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--finetune", default=None, type=str)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path, y.finetune)
