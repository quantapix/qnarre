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

from argparse import ArgumentParser
from os.path import exists
from pathlib import Path
from torch import nn

from transformers import (
    BartModel,
    BartTokenizer,
)
from transformers.utils import logging

from ..config.bart import PreTrained
from ...models.bart import Model, ForSeqClass, ForCondGen

FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": ForSeqClass}


logging.set_verbosity_info()
log = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def make_linear_from_emb(emb):
    s_vocab, emb_size = emb.weight.shape
    lin_layer = nn.Linear(s_vocab, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_checkpoint(src_path, save_path, hf_checkpoint_name=None):
    if not exists(src_path):
        bart = torch.hub.load("pytorch/fairseq", src_path).eval()
    else:
        bart = load_xsum_checkpoint(src_path)
    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = src_path.replace(".", "-")
    cfg = PreTrained.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = (
        BartTokenizer.from_pretrained(hf_checkpoint_name)
        .encode(SAMPLE_TEXT, return_tensors="pt")
        .unsqueeze(0)
    )
    assert torch.eq(tokens, tokens2).all()
    if src_path == "bart.large.mnli":
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        m = ForSeqClass(cfg).eval()
        m.load_state_dict(state_dict)
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        new_model_outputs = m(tokens)[0]
    else:
        state_dict = bart.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = bart.extract_features(tokens)
        if hf_checkpoint_name == "facebook/bart-large":
            m = Model(cfg).eval()
            m.load_state_dict(state_dict)
            new_model_outputs = m(tokens).model[0]
        else:
            m = ForCondGen(cfg).eval()
            m.model.load_state_dict(state_dict)
            if hasattr(m, "lm_head"):
                m.lm_head = make_linear_from_emb(m.model.shared)
            new_model_outputs = m.model(tokens)[0]
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(save_path).mkdir(exist_ok=True)
    m.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", type=str)
    x.add_argument("--cfg_path", default=None, type=str)
    x.add_argument("--save_path", default=None, type=str)
    y = x.parse_args()
    convert_checkpoint(y.src_path, y.save_path, y.cfg_path)
