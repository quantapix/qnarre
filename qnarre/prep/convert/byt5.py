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

from argparse import ArgumentParser
from transformers.utils import logging

from .t5 import load_src_weights
from ..config.t5 import PreTrained
from ...models.t5 import ForCondGen


logging.set_verbosity_info()


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
