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

from tokenizers import pre_tokenizers, processors

from ....tokens.base import AddedToken
from ....tokens.fast import PreTrainedTokenizerFast
from ..roberta import Tokenizer as Roberta


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

VOCAB_MAP = {
    "vocab_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/vocab.json",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/vocab.json",
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/vocab.json",
        "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/vocab.json",
    },
    "merges_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/merges.txt",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/merges.txt",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/merges.txt",
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/merges.txt",
        "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/tokenizer.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/tokenizer.json",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/tokenizer.json",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/tokenizer.json",
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/tokenizer.json",
        "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class Tokenizer(PreTrainedTokenizerFast):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    slow_tokenizer_class = Roberta

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos="<s>",
        eos="</s>",
        sep="</s>",
        cls="<s>",
        unk="<unk>",
        pad="<pad>",
        msk="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kw,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos=bos,
            eos=eos,
            sep=sep,
            cls=cls,
            unk=unk,
            pad=pad,
            msk=msk,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kw,
        )
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        self.add_prefix_space = add_prefix_space
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])
            changes_to_apply = False
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

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

    def _batch_encode_plus(self, *args, **kw):
        is_split_into_words = kw.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        return super()._batch_encode_plus(*args, **kw)

    def _encode_plus(self, *args, **kw):
        is_split_into_words = kw.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        return super()._encode_plus(*args, **kw)

    def save_vocabulary(self, dir, pre=None):
        return tuple(self._tokenizer.model.save(dir, name=pre))

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        y = [self.BOS] + toks_0 + [self.EOS]
        if toks_1 is None:
            return y
        return y + [self.EOS] + toks_1 + [self.EOS]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.SEP]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep + sep + toks_1 + sep) * [0]
