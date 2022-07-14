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
import warnings
from contextlib import contextmanager

from .configuration_rag import RagConfig


class Tokenizer:
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder
        self.generator = generator
        self.current_tokenizer = self.question_encoder

    def save_pretrained(self, dir):
        if os.path.isfile(dir):
            raise ValueError(f"Provided path ({dir}) should be a directory, not a file")
        os.makedirs(dir, exist_ok=True)
        question_encoder_path = os.path.join(dir, "question_encoder_tokenizer")
        generator_path = os.path.join(dir, "generator_tokenizer")
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        # dynamically import AutoTokenizer
        from ..auto.tokenization_auto import AutoTokenizer

        config = kw.pop("config", None)

        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            config=config.question_encoder,
            subfolder="question_encoder_tokenizer",
        )
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kw):
        return self.current_tokenizer(*args, **kw)

    def batch_decode(self, *args, **kw):
        return self.generator.batch_decode(*args, **kw)

    def decode(self, *args, **kw):
        return self.generator.decode(*args, **kw)

    @contextmanager
    def as_target_tokenizer(self):
        self.current_tokenizer = self.generator
        yield
        self.current_tokenizer = self.question_encoder

    def prepare_seq2seq_batch(
        self,
        src_texts,
        tgt_texts=None,
        max_length=None,
        max_target_length=None,
        padding="longest",
        return_tensors=None,
        truncation=True,
        **kw,
    ):
        warnings.warn(
            "`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of 🤗 Transformers. Use the "
            "regular `__call__` method to prepare your inputs and the tokenizer under the `with_target_tokenizer` "
            "context manager to prepare your targets. See the documentation of your specific tokenizer for more "
            "details",
            FutureWarning,
        )
        if max_length is None:
            max_length = self.current_tokenizer.model_max_length
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kw,
        )
        if tgt_texts is None:
            return model_inputs
        with self.as_target_tokenizer():
            if max_target_length is None:
                max_target_length = self.current_tokenizer.model_max_length
            labels = self(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kw,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
