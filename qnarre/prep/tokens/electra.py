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

from .bert import Tokenizer as Bert


VOCAB_FS = {"vocab_file": "vocab.txt"}

VOCAB_MAP = {
    "vocab_file": {
        "google/electra-small-generator": "https://huggingface.co/google/electra-small-generator/resolve/main/vocab.txt",
        "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
        "google/electra-large-generator": "https://huggingface.co/google/electra-large-generator/resolve/main/vocab.txt",
        "google/electra-small-discriminator": "https://huggingface.co/google/electra-small-discriminator/resolve/main/vocab.txt",
        "google/electra-base-discriminator": "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt",
        "google/electra-large-discriminator": "https://huggingface.co/google/electra-large-discriminator/resolve/main/vocab.txt",
    }
}

INPUT_CAPS = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}


class Tokenizer(Bert):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
