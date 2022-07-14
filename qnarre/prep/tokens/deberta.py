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


from ...tokens.utils import AddedToken
from .gpt2 import Tokenizer as GPT2


VOCAB_FS = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

VOCAB_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json",
        "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/vocab.json",
    },
    "merges_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt",
        "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/merges.txt",
    },
}

INPUT_CAPS = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}


class Tokenizer(GPT2):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos="[CLS]",
        eos="[SEP]",
        sep="[SEP]",
        cls="[CLS]",
        unk="[UNK]",
        pad="[PAD]",
        msk="[MASK]",
        add_prefix_space=False,
        **kw,
    ):
        bos = AddedToken(bos, lstrip=False, rstrip=False) if isinstance(bos, str) else bos
        eos = AddedToken(eos, lstrip=False, rstrip=False) if isinstance(eos, str) else eos
        sep = AddedToken(sep, lstrip=False, rstrip=False) if isinstance(sep, str) else sep
        cls = AddedToken(cls, lstrip=False, rstrip=False) if isinstance(cls, str) else cls
        unk = AddedToken(unk, lstrip=False, rstrip=False) if isinstance(unk, str) else unk
        pad = AddedToken(pad, lstrip=False, rstrip=False) if isinstance(pad, str) else pad
        msk = AddedToken(msk, lstrip=True, rstrip=False) if isinstance(msk, str) else msk
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            cls=cls,
            pad=pad,
            msk=msk,
            add_prefix_space=add_prefix_space,
            **kw,
        )

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(self, toks_0, toks_1=None, has_specials=False):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is None:
            return [1] + ([0] * len(toks_0)) + [1]
        return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep + toks_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kw):
        add_prefix_space = kw.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kw)
