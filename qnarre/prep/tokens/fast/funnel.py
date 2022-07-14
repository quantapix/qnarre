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

from .bert import BertFast
from ..funnel import Tokenizer as Funnel


VOCAB_FS = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]

VOCAB_MAP = {
    "vocab_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/vocab.txt",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/vocab.txt",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/vocab.txt",
        "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/vocab.txt",
        "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/vocab.txt",
        "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/vocab.txt",
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/vocab.txt",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/vocab.txt",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/vocab.txt",
        "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/tokenizer.json",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/tokenizer.json",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/tokenizer.json",
        "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/tokenizer.json",
        "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/tokenizer.json",
        "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/tokenizer.json",
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/tokenizer.json",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/tokenizer.json",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/tokenizer.json",
        "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/tokenizer.json",
    },
}
INPUT_CAPS = {f"funnel-transformer/{name}": 512 for name in _model_names}
PRETRAINED_INIT_CONFIGURATION = {
    f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names
}


class Tokenizer(BertFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = Funnel
    cls_token_type_id = 2

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk="<unk>",
        sep="<sep>",
        pad="<pad>",
        cls="<cls>",
        msk="<mask>",
        bos="<s>",
        eos="</s>",
        clean_text=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        wordpieces_prefix="##",
        **kw,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            bos=bos,
            eos=eos,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kw,
        )

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls) * [self.cls_token_type_id] + len(toks_0 + sep) * [0]
        return (
            len(cls) * [self.cls_token_type_id] + len(toks_0 + sep) * [0] + len(toks_1 + sep) * [1]
        )
