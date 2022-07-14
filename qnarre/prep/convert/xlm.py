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
import numpy
import torch

from argparse import ArgumentParser
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.models.xlm.tokenization_xlm import VOCAB_FS
from transformers.utils import logging


logging.set_verbosity_info()


def to_pytorch(src_path, save_path):
    chkpt = torch.load(src_path, map_location="cpu")
    state_dict = chkpt["model"]
    two_levels_state_dict = {}
    for k, v in state_dict.items():
        if "pred_layer" in k:
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict["transformer." + k] = v
    cfg = chkpt["params"]
    cfg = dict(
        (n, v) for n, v in cfg.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray))
    )
    vocab = chkpt["dico_word2id"]
    vocab = dict(
        (s + "</w>" if s.find("@@") == -1 and i > 13 else s.replace("@@", ""), i)
        for s, i in vocab.items()
    )
    w = save_path + "/" + WEIGHTS_NAME
    c = save_path + "/" + CONFIG_NAME
    v = save_path + "/" + VOCAB_FS["vocab_file"]
    print(f"Saving to: {w}")
    torch.save(two_levels_state_dict, w)
    print(f"Saving config to: {c}")
    with open(c, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
    print(f"Saving vocab to: {v}")
    with open(v, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, indent=2) + "\n")


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.save_path)
