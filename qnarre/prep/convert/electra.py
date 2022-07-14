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
import re
import numpy as np
import tensorflow as tf
import torch

from argparse import ArgumentParser
from transformers.utils import logging

from ..config.electra import PreTrained
from ...models.electra import ForPreTraining, ForMasked


logging.set_verbosity_info()

log = logging.get_logger(__name__)


def load_src_weights(model, src_path, discriminator_or_generator="discriminator"):
    src_path = os.path.abspath(src_path)
    log.info(f"Loading from: {src_path}")
    init_vars = tf.train.list_variables(src_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(src_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        original_name = name
        try:
            if isinstance(model, ForMasked):
                name = name.replace("electra/embeddings/", "generator/embeddings/")
            if discriminator_or_generator == "generator":
                name = name.replace("electra/", "discriminator/")
                name = name.replace("generator/", "electra/")
            name = name.replace("dense_1", "dense_prediction")
            name = name.replace("generator_predictions/output_bias", "generator_lm_head/bias")
            name = name.split("/")
            if any(n in ["global_step", "temperature"] for n in name):
                log.info(f"Skipping {original_name}")
                continue
            p = model
            for s in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", s):
                    scopes = re.split(r"_(\d+)", s)
                else:
                    scopes = [s]
                if scopes[0] == "kernel" or scopes[0] == "gamma":
                    p = getattr(p, "weight")
                elif scopes[0] == "output_bias" or scopes[0] == "beta":
                    p = getattr(p, "bias")
                elif scopes[0] == "output_weights":
                    p = getattr(p, "weight")
                elif scopes[0] == "squad":
                    p = getattr(p, "classifier")
                else:
                    p = getattr(p, scopes[0])
                if len(scopes) >= 2:
                    num = int(scopes[1])
                    p = p[num]
            if s.endswith("_embeddings"):
                p = getattr(p, "weight")
            elif s == "kernel":
                array = np.transpose(array)
            assert p.shape == array.shape
            p.data = torch.from_numpy(array)
        except AttributeError as e:
            print(f"Skipping {original_name}", name, e)
            continue
    return model


def to_pytorch(src_path, cfg_path, save_path, discriminator_or_generator):
    cfg = PreTrained.from_json_file(cfg_path)
    print(f"Building from config: {cfg}")
    if discriminator_or_generator == "discriminator":
        m = ForPreTraining(cfg)
    else:
        assert discriminator_or_generator == "generator"
        m = ForMasked(cfg)
    load_src_weights(m, src_path, discriminator_or_generator)
    print(f"Saving to: {save_path}")
    torch.save(m.state_dict(), save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--cfg_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--discriminator_or_generator", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.cfg_path, y.save_path, y.discriminator_or_generator)
