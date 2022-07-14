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
import pickle
import sys
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath, join
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.models.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FS
from transformers.utils import logging

from ..config.transfo_xl import PreTrained
from ...models.transfo_xl import LMHead


logging.set_verbosity_info()

log = logging.get_logger(__name__)

data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules["data_utils"] = data_utils
sys.modules["vocabulary"] = data_utils


def build_map(model, cfg):
    tf_to_pt_map = {}
    if hasattr(model, "transformer"):
        tf_to_pt_map.update(
            {
                "transformer/adaptive_softmax/cutoff_0/cluster_W": model.crit.cluster_weight,
                "transformer/adaptive_softmax/cutoff_0/cluster_b": model.crit.cluster_bias,
            }
        )
        for i, (out_l, proj_l, tie_proj) in enumerate(
            zip(model.crit.out_layers, model.crit.out_projs, cfg.tie_projs)
        ):
            layer_str = f"transformer/adaptive_softmax/cutoff_{i}/"
            if cfg.tie_word_embeds:
                tf_to_pt_map.update({layer_str + "b": out_l.bias})
            else:
                raise NotImplementedError
                # I don't think this is implemented in the TF code
                tf_to_pt_map.update(
                    {layer_str + "lookup_table": out_l.weight, layer_str + "b": out_l.bias}
                )
            if not tie_proj:
                tf_to_pt_map.update({layer_str + "proj": proj_l})
        model = model.transformer
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = f"transformer/adaptive_embed/cutoff_{i}/"
        tf_to_pt_map.update(
            {layer_str + "lookup_table": embed_l.weight, layer_str + "proj_W": proj_l}
        )
    for i, b in enumerate(model.layers):
        layer_str = f"transformer/layer_{i}/"
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.dec_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.dec_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.dec_attn.o_net.weight,
                layer_str + "rel_attn/qkv/kernel": b.dec_attn.qkv_net.weight,
                layer_str + "rel_attn/r/kernel": b.dec_attn.r_net.weight,
                layer_str + "ff/LayerNorm/gamma": b.pos_ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.pos_ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.pos_ff.CoreNet[0].weight,
                layer_str + "ff/layer_1/bias": b.pos_ff.CoreNet[0].bias,
                layer_str + "ff/layer_2/kernel": b.pos_ff.CoreNet[3].weight,
                layer_str + "ff/layer_2/bias": b.pos_ff.CoreNet[3].bias,
            }
        )
    if cfg.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({"transformer/r_r_bias": r_r_list, "transformer/r_w_bias": r_w_list})
    return tf_to_pt_map


def load_src_weights(model, cfg, src_path):
    tf_to_pt_map = build_map(model, cfg)
    init_vars = tf.train.list_variables(src_path)
    tf_weights = {}
    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(src_path, name)
        tf_weights[name] = array
    for name, p in tf_to_pt_map.items():
        assert name in tf_weights
        array = tf_weights[name]
        if "kernel" in name or "proj" in name:
            array = np.transpose(array)
        if ("r_r_bias" in name or "r_w_bias" in name) and len(p) > 1:
            assert len(p) == array.shape[0]
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


def to_pytorch(src_path, cfg_path, save_path, ds_path):
    if ds_path:
        with open(ds_path, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        v = save_path + "/" + VOCAB_FS["pretrained_vocab_file"]
        print(f"Saving vocab to: {v}")
        torch.save(corpus.vocab.__dict__, v)
        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop("vocab", None)
        d = save_path + "/" + CORPUS_NAME
        print(f"Saving dataset to: {d}")
        torch.save(corpus_dict_no_vocab, d)
    if src_path:
        cfg_path = abspath(cfg_path)
        cfg = PreTrained() if cfg_path == "" else PreTrained.from_json_file(cfg_path)
        print(f"Building from config: {cfg}")
        m = LMHead(cfg)
        src_path = abspath(src_path)
        m = load_src_weights(m, cfg, src_path)
        w = join(save_path, WEIGHTS_NAME)
        print(f"Saving to: {abspath(w)}")
        torch.save(m.state_dict(), w)
        c = join(save_path, CONFIG_NAME)
        print(f"Saving config to: {abspath(c)}")
        with open(c, "w", encoding="utf-8") as f:
            f.write(cfg.to_json_string())


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default="", type=str)
    x.add_argument("--cfg_path", default="", type=str)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--ds_path", default="", type=str)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path, y.ds_path)
