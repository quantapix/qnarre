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

from tokenizers import normalizers

from ....tokens.utils import AddedToken
from ....tokens.fast import PreTrainedTokenizerFast
from ..mpnet import Tokenizer as MPNet


VOCAB_FS = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "microsoft/mpnet-base": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},
}


class MPNetTokenizerFast(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    slow_tokenizer_class = MPNet
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        bos="<s>",
        eos="</s>",
        sep="</s>",
        cls="<s>",
        unk="[UNK]",
        pad="<pad>",
        msk="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kw,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos=bos,
            eos=eos,
            sep=sep,
            cls=cls,
            unk=unk,
            pad=pad,
            msk=msk,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kw,
        )
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)
        self.do_lower_case = do_lower_case

    @property
    def msk(self):
        if self._mask_token is None and self.verbose:
            logger.error("Using msk, but it is not set yet.")
            return None
        return str(self._mask_token)

    @msk.setter
    def msk(self, value):
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        y = [self.BOS] + toks_0 + [self.EOS]
        if toks_1 is None:
            return y
        return y + [self.EOS] + toks_1 + [self.EOS]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep + sep + toks_1 + sep) * [0]

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))
