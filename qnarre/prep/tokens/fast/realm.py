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

from ....tokens.base import BatchEncoding
from ....tokens.fast import PreTrainedTokenizerFast
from ....tokens.utils import PaddingStrategy
from ..realm import Tokenizer as Realm


VOCAB_FS = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

VOCAB_MAP = {
    "vocab_file": {
        "google/realm-cc-news-pretrained-embedder": "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/vocab.txt",
        "google/realm-cc-news-pretrained-encoder": "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/vocab.txt",
        "google/realm-cc-news-pretrained-scorer": "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/vocab.txt",
        "google/realm-cc-news-pretrained-openqa": "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/vocab.txt",
        "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/vocab.txt",
        "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "google/realm-cc-news-pretrained-embedder": "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/tokenizer.jsont",
        "google/realm-cc-news-pretrained-encoder": "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/tokenizer.json",
        "google/realm-cc-news-pretrained-scorer": "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/tokenizer.json",
        "google/realm-cc-news-pretrained-openqa": "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/tokenizer.json",
        "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/tokenizer.json",
        "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/tokenizer.json",
        "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/tokenizer.json",
        "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "google/realm-cc-news-pretrained-embedder": 512,
    "google/realm-cc-news-pretrained-encoder": 512,
    "google/realm-cc-news-pretrained-scorer": 512,
    "google/realm-cc-news-pretrained-openqa": 512,
    "google/realm-orqa-nq-openqa": 512,
    "google/realm-orqa-nq-reader": 512,
    "google/realm-orqa-wq-openqa": 512,
    "google/realm-orqa-wq-reader": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "google/realm-cc-news-pretrained-embedder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-encoder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-scorer": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-reader": {"do_lower_case": True},
    "google/realm-orqa-wq-openqa": {"do_lower_case": True},
    "google/realm-orqa-wq-reader": {"do_lower_case": True},
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    slow_tokenizer_class = Realm

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk="[UNK]",
        sep="[SEP]",
        pad="[PAD]",
        cls="[CLS]",
        msk="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
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
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kw,
        )
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars)
            != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)
        self.do_lower_case = do_lower_case

    def batch_encode_candidates(self, text, **kw):
        kw["padding"] = PaddingStrategy.MAX_LENGTH
        batch_text = text
        batch_text_pair = kw.pop("text_pair", None)
        return_tensors = kw.pop("return_tensors", None)
        output_data = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        for i, candidate_text in enumerate(batch_text):
            if batch_text_pair is not None:
                candidate_text_pair = batch_text_pair[i]
            else:
                candidate_text_pair = None
            encoded_candidates = super().__call__(
                candidate_text, candidate_text_pair, return_tensors=None, **kw
            )
            encoded_input_ids = encoded_candidates.get("input_ids")
            encoded_attention_mask = encoded_candidates.get("attention_mask")
            encoded_token_type_ids = encoded_candidates.get("token_type_ids")
            if encoded_input_ids is not None:
                output_data["input_ids"].append(encoded_input_ids)
            if encoded_attention_mask is not None:
                output_data["attention_mask"].append(encoded_attention_mask)
            if encoded_token_type_ids is not None:
                output_data["token_type_ids"].append(encoded_token_type_ids)
        output_data = dict((key, item) for key, item in output_data.items() if len(item) != 0)
        return BatchEncoding(output_data, tensor_type=return_tensors)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        y = [self.cls_token_id] + toks_0 + [self.sep_token_id]
        if toks_1:
            y += toks_1 + [self.sep_token_id]
        return y

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))
