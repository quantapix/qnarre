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

from ....tokens.utils import PaddingStrategy
from .bart import Tokenizer as BartFast
from ..led import Tokenizer as LED


VOCAB_MAP = {
    "vocab_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

INPUT_CAPS = {
    "allenai/led-base-16384": 16384,
}


class Tokenizer(BartFast):
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS
    slow_tokenizer_class = LED

    def _pad(
        self,
        encoded_inputs,
        max_length=None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ):
        encoded_inputs = super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            required_input = encoded_inputs[self.model_input_names[0]]
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)
            if needs_to_be_padded:
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])
                if self.padding_side == "right":
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                else:
                    assert self.padding_side == "left"
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
        return encoded_inputs
