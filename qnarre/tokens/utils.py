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

import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict

from .file_utils import PaddingStrategy
from .base import (
    AddedToken,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import logging


log = logging.get_logger(__name__)

# Slow tokenizers are saved in a vocabulary plus three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class Trie:
    def __init__(self):
        self.data = {}

    def add(self, word):
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text):
        states = OrderedDict()
        offsets = [0]
        skip = 0
        for current, current_char in enumerate(text):
            if skip and current < skip:
                continue
            to_remove = set()
            reset = False
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            break
                        elif lookstart < start:
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                log.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list, new_token):
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)


class PreTrainedTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}
        self.unique_no_split_tokens = []
        self.tokens_trie = Trie()
        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self):
        return False

    @property
    def s_vocab(self):
        raise NotImplementedError

    def get_added_vocab(self):
        return self.added_tokens_encoder

    def __len__(self):
        return self.s_vocab + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens, special_tokens=False):
        new_tokens = [str(tok) for tok in new_tokens]
        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    log.info(f"Adding {token} to the vocabulary")
        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(
                    set(self.unique_no_split_tokens).union(set(new_tokens))
                )
        else:
            # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(
                    set(self.unique_no_split_tokens).union(set(tokens_to_add))
                )
        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if (
                hasattr(self, "do_lower_case")
                and self.do_lower_case
                and token not in self.all_special_tokens
            ):
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def num_special_tokens_to_add(self, pair=False):
        toks_0 = []
        toks_1 = []
        return len(self.build_inputs_with_special_tokens(toks_0, toks_1 if pair else None))

    def tokenize(self, text: TextInput, **kw):
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        )

        text, kw = self.prepare_for_tokenization(text, **kw)

        if kw:
            log.warning(f"Keyword arguments {kw} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok)
                for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kw):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_len=None,
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
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kw)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(
                            *(self.tokenize(t, is_split_into_words=True, **kw) for t in text)
                        )
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_len=max_len,
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
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_len=None,
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
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kw)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(
                            *(self.tokenize(t, is_split_into_words=True, **kw) for t in text)
                        )
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_len=max_len,
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

    def _batch_prepare_for_model(
        self,
        batch_ids_pairs,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_len=None,
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
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_len=max_len,
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
            max_len=max_len,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kw):
        return (text, kw)

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            if toks_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        return [0] * ((len(toks_1) if toks_1 else 0) + len(toks_0))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        ...

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        ...

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def _decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        spaces_between_special_tokens=True,
        **kw,
    ):
        self._decode_use_source_tokenizer = kw.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
