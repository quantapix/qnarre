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

from .roberta import Tokenizer as Roberta


VOCAB_FS = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

VOCAB_MAP = {
    "vocab_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
        "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/vocab.json",
        "allenai/longformer-large-4096-finetuned-triviaqa": "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/vocab.json",
        "allenai/longformer-base-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/vocab.json",
        "allenai/longformer-large-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
        "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/merges.txt",
        "allenai/longformer-large-4096-finetuned-triviaqa": "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/merges.txt",
        "allenai/longformer-base-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/merges.txt",
        "allenai/longformer-large-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/merges.txt",
    },
}

INPUT_CAPS = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
    "allenai/longformer-large-4096-extra.pos.embd.only": 4096,
}


class Tokenizer(Roberta):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
