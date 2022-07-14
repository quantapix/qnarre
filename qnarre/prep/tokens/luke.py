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

import itertools
import json
import os

import numpy as np

from .roberta import Tokenizer as Roberta
from ...tokens.utils import (
    AddedToken,
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
    _is_tensorflow,
    _is_torch,
    to_py_obj,
)
from ...tokens.utils import is_tf_available, is_torch_available


VOCAB_FS = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "entity_vocab_file": "entity_vocab.json",
}

VOCAB_MAP = {
    "vocab_file": {
        "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/vocab.json",
        "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/vocab.json",
    },
    "merges_file": {
        "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/merges.txt",
        "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/merges.txt",
    },
    "entity_vocab_file": {
        "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/entity_vocab.json",
        "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/entity_vocab.json",
    },
}

INPUT_CAPS = {
    "studio-ousia/luke-base": 512,
    "studio-ousia/luke-large": 512,
}


class Tokenizer(Roberta):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    input_caps = INPUT_CAPS

    def __init__(
        self,
        vocab_file,
        merges_file,
        entity_vocab_file,
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        **kw,
    ):
        entity_token_1 = (
            AddedToken(entity_token_1, lstrip=False, rstrip=False)
            if isinstance(entity_token_1, str)
            else entity_token_1
        )
        entity_token_2 = (
            AddedToken(entity_token_2, lstrip=False, rstrip=False)
            if isinstance(entity_token_2, str)
            else entity_token_2
        )
        kw["additional_special_tokens"] = kw.get("additional_special_tokens", [])
        kw["additional_special_tokens"] += [entity_token_1, entity_token_2]
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            task=task,
            max_entity_length=32,
            max_mention_length=30,
            entity_token_1="<ent>",
            entity_token_2="<ent2>",
            entity_unk_token=entity_unk_token,
            entity_pad_token=entity_pad_token,
            entity_mask_token=entity_mask_token,
            entity_mask2_token=entity_mask2_token,
            **kw,
        )
        with open(entity_vocab_file, encoding="utf-8") as entity_vocab_handle:
            self.entity_vocab = json.load(entity_vocab_handle)
        for entity_special_token in [
            entity_unk_token,
            entity_pad_token,
            entity_mask_token,
            entity_mask2_token,
        ]:
            if entity_special_token not in self.entity_vocab:
                raise ValueError(
                    f"Specified entity special token ``{entity_special_token}`` is not found in entity_vocab. "
                    f"Probably an incorrect entity vocab file is loaded: {entity_vocab_file}."
                )
        self.entity_unk_token_id = self.entity_vocab[entity_unk_token]
        self.entity_pad_token_id = self.entity_vocab[entity_pad_token]
        self.entity_mask_token_id = self.entity_vocab[entity_mask_token]
        self.entity_mask2_token_id = self.entity_vocab[entity_mask2_token]
        self.task = task
        if task is None or task == "entity_span_classification":
            self.max_entity_length = max_entity_length
        elif task == "entity_classification":
            self.max_entity_length = 1
        elif task == "entity_pair_classification":
            self.max_entity_length = 2
        else:
            raise ValueError(
                f"Task {task} not supported. Select task from ['entity_classification', 'entity_pair_classification', 'entity_span_classification'] only."
            )
        self.max_mention_length = max_mention_length

    def __call__(
        self,
        text,
        text_pair=None,
        entity_spans=None,
        entity_spans_pair=None,
        entities=None,
        entities_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        max_entity_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kw,
    ):
        is_valid_single_text = isinstance(text, str)
        is_valid_batch_text = isinstance(text, (list, tuple)) and (
            len(text) == 0 or (isinstance(text[0], str))
        )
        if not (is_valid_single_text or is_valid_batch_text):
            raise ValueError(
                "text input must be of type `str` (single example) or `List[str]` (batch)."
            )
        is_valid_single_text_pair = isinstance(text_pair, str)
        is_valid_batch_text_pair = isinstance(text_pair, (list, tuple)) and (
            len(text_pair) == 0 or isinstance(text_pair[0], str)
        )
        if not (text_pair is None or is_valid_single_text_pair or is_valid_batch_text_pair):
            raise ValueError(
                "text_pair input must be of type `str` (single example) or `List[str]` (batch)."
            )
        is_batched = bool(isinstance(text, (list, tuple)))
        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            if entities is None:
                batch_entities_or_entities_pairs = None
            else:
                batch_entities_or_entities_pairs = (
                    list(zip(entities, entities_pair)) if entities_pair is not None else entities
                )
            if entity_spans is None:
                batch_entity_spans_or_entity_spans_pairs = None
            else:
                batch_entity_spans_or_entity_spans_pairs = (
                    list(zip(entity_spans, entity_spans_pair))
                    if entity_spans_pair is not None
                    else entity_spans
                )
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                batch_entity_spans_or_entity_spans_pairs=batch_entity_spans_or_entity_spans_pairs,
                batch_entities_or_entities_pairs=batch_entities_or_entities_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kw,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                entity_spans=entity_spans,
                entity_spans_pair=entity_spans_pair,
                entities=entities,
                entities_pair=entities_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kw,
            )

    def _encode_plus(
        self,
        text,
        text_pair=None,
        entity_spans=None,
        entity_spans_pair=None,
        entities=None,
        entities_pair=None,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_length=None,
        max_entity_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kw,
    ):

        if return_offsets_mapping:
            raise NotImplementedError()
        if is_split_into_words:
            raise NotImplementedError()
        (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair,
            **kw,
        )
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=first_entity_ids,
            pair_entity_ids=second_entity_ids,
            entity_token_spans=first_entity_token_spans,
            pair_entity_token_spans=second_entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        batch_entity_spans_or_entity_spans_pairs=None,
        batch_entities_or_entities_pairs=None,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_length=None,
        max_entity_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kw,
    ):
        if return_offsets_mapping:
            raise NotImplementedError()
        if is_split_into_words:
            raise NotImplementedError()
        input_ids = []
        entity_ids = []
        entity_token_spans = []
        for index, text_or_text_pair in enumerate(batch_text_or_text_pairs):
            if not isinstance(text_or_text_pair, (list, tuple)):
                text, text_pair = text_or_text_pair, None
            else:
                text, text_pair = text_or_text_pair

            entities, entities_pair = None, None
            if batch_entities_or_entities_pairs is not None:
                entities_or_entities_pairs = batch_entities_or_entities_pairs[index]
                if entities_or_entities_pairs:
                    if isinstance(entities_or_entities_pairs[0], str):
                        entities, entities_pair = entities_or_entities_pairs, None
                    else:
                        entities, entities_pair = entities_or_entities_pairs

            entity_spans, entity_spans_pair = None, None
            if batch_entity_spans_or_entity_spans_pairs is not None:
                entity_spans_or_entity_spans_pairs = batch_entity_spans_or_entity_spans_pairs[index]
                if len(entity_spans_or_entity_spans_pairs) > 0 and isinstance(
                    entity_spans_or_entity_spans_pairs[0], list
                ):
                    entity_spans, entity_spans_pair = entity_spans_or_entity_spans_pairs
                else:
                    entity_spans, entity_spans_pair = entity_spans_or_entity_spans_pairs, None

            (
                first_ids,
                second_ids,
                first_entity_ids,
                second_entity_ids,
                first_entity_token_spans,
                second_entity_token_spans,
            ) = self._create_input_sequence(
                text=text,
                text_pair=text_pair,
                entities=entities,
                entities_pair=entities_pair,
                entity_spans=entity_spans,
                entity_spans_pair=entity_spans_pair,
                **kw,
            )
            input_ids.append((first_ids, second_ids))
            entity_ids.append((first_entity_ids, second_entity_ids))
            entity_token_spans.append((first_entity_token_spans, second_entity_token_spans))
        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            batch_entity_ids_pairs=entity_ids,
            batch_entity_token_spans_pairs=entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )
        return BatchEncoding(batch_outputs)

    def _check_entity_input_format(self, entities, entity_spans):
        if not isinstance(entity_spans, list):
            raise ValueError("entity_spans should be given as a list")
        elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):
            raise ValueError(
                "entity_spans should be given as a list of tuples "
                "containing the start and end character indices"
            )
        if entities is not None:
            if not isinstance(entities, list):
                raise ValueError("If you specify entities, they should be given as a list")
            if len(entities) > 0 and not isinstance(entities[0], str):
                raise ValueError(
                    "If you specify entities, they should be given as a list of entity names"
                )
            if len(entities) != len(entity_spans):
                raise ValueError(
                    "If you specify entities, entities and entity_spans must be the same length"
                )

    def _create_input_sequence(
        self,
        text,
        text_pair=None,
        entities=None,
        entities_pair=None,
        entity_spans=None,
        entity_spans_pair=None,
        **kw,
    ):
        def get_input_ids(text):
            tokens = self.tokenize(text, **kw)
            return self.convert_tokens_to_ids(tokens)

        def get_input_ids_and_entity_token_spans(text, entity_spans):
            if entity_spans is None:
                return get_input_ids(text), None
            cur = 0
            input_ids = []
            entity_token_spans = [None] * len(entity_spans)
            split_char_positions = sorted(frozenset(itertools.chain(*entity_spans)))
            char_pos2token_pos = {}
            for split_char_position in split_char_positions:
                orig_split_char_position = split_char_position
                if split_char_position > 0 and text[split_char_position - 1] == " ":
                    split_char_position -= 1
                if cur != split_char_position:
                    input_ids += get_input_ids(text[cur:split_char_position])
                    cur = split_char_position
                char_pos2token_pos[orig_split_char_position] = len(input_ids)
            input_ids += get_input_ids(text[cur:])
            entity_token_spans = [
                (char_pos2token_pos[char_start], char_pos2token_pos[char_end])
                for char_start, char_end in entity_spans
            ]
            return input_ids, entity_token_spans

        first_ids, second_ids = None, None
        first_entity_ids, second_entity_ids = None, None
        first_entity_token_spans, second_entity_token_spans = None, None
        if self.task is None:
            if entity_spans is None:
                first_ids = get_input_ids(text)
            else:
                self._check_entity_input_format(entities, entity_spans)

                first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(
                    text, entity_spans
                )
                if entities is None:
                    first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)
                else:
                    first_entity_ids = [
                        self.entity_vocab.get(entity, self.entity_unk_token_id)
                        for entity in entities
                    ]
            if text_pair is not None:
                if entity_spans_pair is None:
                    second_ids = get_input_ids(text_pair)
                else:
                    self._check_entity_input_format(entities_pair, entity_spans_pair)

                    second_ids, second_entity_token_spans = get_input_ids_and_entity_token_spans(
                        text_pair, entity_spans_pair
                    )
                    if entities_pair is None:
                        second_entity_ids = [self.entity_mask_token_id] * len(entity_spans_pair)
                    else:
                        second_entity_ids = [
                            self.entity_vocab.get(entity, self.entity_unk_token_id)
                            for entity in entities_pair
                        ]
        elif self.task == "entity_classification":
            if not (
                isinstance(entity_spans, list)
                and len(entity_spans) == 1
                and isinstance(entity_spans[0], tuple)
            ):
                raise ValueError(
                    "Entity spans should be a list containing a single tuple "
                    "containing the start and end character indices of an entity"
                )
            first_entity_ids = [self.entity_mask_token_id]
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(
                text, entity_spans
            )
            entity_token_start, entity_token_end = first_entity_token_spans[0]
            first_ids = (
                first_ids[:entity_token_end]
                + [self.additional_special_tokens_ids[0]]
                + first_ids[entity_token_end:]
            )
            first_ids = (
                first_ids[:entity_token_start]
                + [self.additional_special_tokens_ids[0]]
                + first_ids[entity_token_start:]
            )
            first_entity_token_spans = [(entity_token_start, entity_token_end + 2)]

        elif self.task == "entity_pair_classification":
            if not (
                isinstance(entity_spans, list)
                and len(entity_spans) == 2
                and isinstance(entity_spans[0], tuple)
                and isinstance(entity_spans[1], tuple)
            ):
                raise ValueError(
                    "Entity spans should be provided as a list of two tuples, "
                    "each tuple containing the start and end character indices of an entity"
                )
            head_span, tail_span = entity_spans
            first_entity_ids = [self.entity_mask_token_id, self.entity_mask2_token_id]
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(
                text, entity_spans
            )
            head_token_span, tail_token_span = first_entity_token_spans
            token_span_with_special_token_ids = [
                (head_token_span, self.additional_special_tokens_ids[0]),
                (tail_token_span, self.additional_special_tokens_ids[1]),
            ]
            if head_token_span[0] < tail_token_span[0]:
                first_entity_token_spans[0] = (head_token_span[0], head_token_span[1] + 2)
                first_entity_token_spans[1] = (tail_token_span[0] + 2, tail_token_span[1] + 4)
                token_span_with_special_token_ids = reversed(token_span_with_special_token_ids)
            else:
                first_entity_token_spans[0] = (head_token_span[0] + 2, head_token_span[1] + 4)
                first_entity_token_spans[1] = (tail_token_span[0], tail_token_span[1] + 2)

            for (
                entity_token_start,
                entity_token_end,
            ), special_token_id in token_span_with_special_token_ids:
                first_ids = (
                    first_ids[:entity_token_end] + [special_token_id] + first_ids[entity_token_end:]
                )
                first_ids = (
                    first_ids[:entity_token_start]
                    + [special_token_id]
                    + first_ids[entity_token_start:]
                )
        elif self.task == "entity_span_classification":
            if not (
                isinstance(entity_spans, list)
                and len(entity_spans) > 0
                and isinstance(entity_spans[0], tuple)
            ):
                raise ValueError(
                    "Entity spans should be provided as a list of tuples, "
                    "each tuple containing the start and end character indices of an entity"
                )
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(
                text, entity_spans
            )
            first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)
        else:
            raise ValueError(f"Task {self.task} not supported")

        return (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        )

    def _batch_prepare_for_model(
        self,
        batch_ids_pairs,
        batch_entity_ids_pairs,
        batch_entity_token_spans_pairs,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_length=None,
        max_entity_length=None,
        stride=0,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_length=False,
        verbose=True,
    ):
        batch_outputs = {}
        for input_ids, entity_ids, entity_token_span_pairs in zip(
            batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
        ):
            first_ids, second_ids = input_ids
            first_entity_ids, second_entity_ids = entity_ids
            first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids=first_entity_ids,
                pair_entity_ids=second_entity_ids,
                entity_token_spans=first_entity_token_spans,
                pair_entity_token_spans=second_entity_token_spans,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        entity_ids=None,
        pair_entity_ids=None,
        entity_token_spans=None,
        pair_entity_token_spans=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        max_entity_length=None,
        stride=0,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        prepend_batch_axis=False,
        **kw,
    ):
        (
            padding_strategy,
            truncation_strategy,
            max_length,
            kw,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kw,
        )

        # Compute lengths
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )
        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        encoded_inputs = {}
        total_len = (
            len_ids
            + len_pair_ids
            + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        )

        # Truncation: Handle max sequence length and max_entity_length
        overflowing_tokens = []
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and max_length
            and total_len > max_length
        ):
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            entity_token_offset = 1  # 1 * <s> token
            pair_entity_token_offset = len(ids) + 3  # 1 * <s> token & 2 * <sep> tokens
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            entity_token_offset = 0
            pair_entity_token_offset = len(ids)
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)
        if not max_entity_length:
            max_entity_length = self.max_entity_length

        if entity_ids is not None:
            total_entity_len = 0
            num_invalid_entities = 0
            valid_entity_ids = [
                ent_id
                for ent_id, span in zip(entity_ids, entity_token_spans)
                if span[1] <= len(ids)
            ]
            valid_entity_token_spans = [span for span in entity_token_spans if span[1] <= len(ids)]

            total_entity_len += len(valid_entity_ids)
            num_invalid_entities += len(entity_ids) - len(valid_entity_ids)

            valid_pair_entity_ids, valid_pair_entity_token_spans = None, None
            if pair_entity_ids is not None:
                valid_pair_entity_ids = [
                    ent_id
                    for ent_id, span in zip(pair_entity_ids, pair_entity_token_spans)
                    if span[1] <= len(pair_ids)
                ]
                valid_pair_entity_token_spans = [
                    span for span in pair_entity_token_spans if span[1] <= len(pair_ids)
                ]
                total_entity_len += len(valid_pair_entity_ids)
                num_invalid_entities += len(pair_entity_ids) - len(valid_pair_entity_ids)

            if num_invalid_entities != 0:
                logger.warning(
                    f"{num_invalid_entities} entities are ignored because their entity spans are invalid due to the truncation of input tokens"
                )

            if (
                truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                and total_entity_len > max_entity_length
            ):
                (
                    valid_entity_ids,
                    valid_pair_entity_ids,
                    overflowing_entities,
                ) = self.truncate_sequences(
                    valid_entity_ids,
                    pair_ids=valid_pair_entity_ids,
                    num_tokens_to_remove=total_entity_len - max_entity_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )
                valid_entity_token_spans = valid_entity_token_spans[: len(valid_entity_ids)]
                if valid_pair_entity_token_spans is not None:
                    valid_pair_entity_token_spans = valid_pair_entity_token_spans[
                        : len(valid_pair_entity_ids)
                    ]
            if return_overflowing_tokens:
                encoded_inputs["overflowing_entities"] = overflowing_entities
                encoded_inputs["num_truncated_entities"] = total_entity_len - max_entity_length
            final_entity_ids = (
                valid_entity_ids + valid_pair_entity_ids
                if valid_pair_entity_ids
                else valid_entity_ids
            )
            encoded_inputs["entity_ids"] = list(final_entity_ids)
            entity_position_ids = []
            entity_start_positions = []
            entity_end_positions = []
            for (token_spans, offset) in (
                (valid_entity_token_spans, entity_token_offset),
                (valid_pair_entity_token_spans, pair_entity_token_offset),
            ):
                if token_spans is not None:
                    for start, end in token_spans:
                        start += offset
                        end += offset
                        position_ids = list(range(start, end))[: self.max_mention_length]
                        position_ids += [-1] * (self.max_mention_length - end + start)
                        entity_position_ids.append(position_ids)
                        entity_start_positions.append(start)
                        entity_end_positions.append(end - 1)

            encoded_inputs["entity_position_ids"] = entity_position_ids
            if self.task == "entity_span_classification":
                encoded_inputs["entity_start_positions"] = entity_start_positions
                encoded_inputs["entity_end_positions"] = entity_end_positions

            if return_token_type_ids:
                encoded_inputs["entity_token_type_ids"] = [0] * len(encoded_inputs["entity_ids"])
        self._eventual_warn_about_too_long_sequence(
            encoded_inputs["input_ids"], max_length, verbose
        )
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )
        return batch_outputs

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        max_entity_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=None,
        return_tensors=None,
        verbose=True,
    ):
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], (dict, BatchEncoding)
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )
        required_input = encoded_inputs[self.model_input_names[0]]
        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs
        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )
            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )
        if max_entity_length is None:
            max_entity_length = self.max_entity_length
        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        if any(len(v) != batch_size for v in encoded_inputs.values()):
            raise ValueError(
                "Some items in the output dictionary have a different batch size than others."
            )

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            max_entity_length = (
                max(len(inputs) for inputs in encoded_inputs["entity_ids"])
                if "entity_ids" in encoded_inputs
                else 0
            )
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs,
        max_length=None,
        max_entity_length=None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ):
        entities_provided = bool("entity_ids" in encoded_inputs)
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])
            if entities_provided:
                max_entity_length = len(encoded_inputs["entity_ids"])
        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if (
            entities_provided
            and max_entity_length is not None
            and pad_to_multiple_of is not None
            and (max_entity_length % pad_to_multiple_of != 0)
        ):
            max_entity_length = ((max_entity_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            len(encoded_inputs["input_ids"]) != max_length
            or (entities_provided and len(encoded_inputs["entity_ids"]) != max_entity_length)
        )
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
        if (
            entities_provided
            and return_attention_mask
            and "entity_attention_mask" not in encoded_inputs
        ):
            encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"])
        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"])
            if entities_provided:
                entity_difference = max_entity_length - len(encoded_inputs["entity_ids"])
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"] + [0] * difference
                    )
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = (
                            encoded_inputs["entity_attention_mask"] + [0] * entity_difference
                        )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [0] * difference
                    )
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = (
                            encoded_inputs["entity_token_type_ids"] + [0] * entity_difference
                        )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.PAD] * difference
                if entities_provided:
                    encoded_inputs["entity_ids"] = (
                        encoded_inputs["entity_ids"]
                        + [self.entity_pad_token_id] * entity_difference
                    )
                    encoded_inputs["entity_position_ids"] = (
                        encoded_inputs["entity_position_ids"]
                        + [[-1] * self.max_mention_length] * entity_difference
                    )
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = (
                            encoded_inputs["entity_start_positions"] + [0] * entity_difference
                        )
                        encoded_inputs["entity_end_positions"] = (
                            encoded_inputs["entity_end_positions"] + [0] * entity_difference
                        )
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs[
                        "attention_mask"
                    ]
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = [
                            0
                        ] * entity_difference + encoded_inputs["entity_attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [0] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = [
                            0
                        ] * entity_difference + encoded_inputs["entity_token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs[
                        "special_tokens_mask"
                    ]
                encoded_inputs["input_ids"] = [self.PAD] * difference + encoded_inputs["input_ids"]
                if entities_provided:
                    encoded_inputs["entity_ids"] = [
                        self.entity_pad_token_id
                    ] * entity_difference + encoded_inputs["entity_ids"]
                    encoded_inputs["entity_position_ids"] = [
                        [-1] * self.max_mention_length
                    ] * entity_difference + encoded_inputs["entity_position_ids"]
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = [
                            0
                        ] * entity_difference + encoded_inputs["entity_start_positions"]
                        encoded_inputs["entity_end_positions"] = [
                            0
                        ] * entity_difference + encoded_inputs["entity_end_positions"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

    def save_vocabulary(self, dir, pre=None):
        vocab_file, merge_file = super().save_vocabulary(dir, pre)
        entity_vocab_file = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["entity_vocab_file"],
        )
        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, ensure_ascii=False))
        return vocab_file, merge_file, entity_vocab_file
