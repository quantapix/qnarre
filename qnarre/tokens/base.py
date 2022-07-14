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

import copy
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from requests import HTTPError

# from .dynamic_module_utils import custom_object_save
from ..core import (
    EntryNotFoundError,
    ExplicitEnum,
    PaddingStrategy,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    TensorType,
    _is_numpy,
    _is_tensorflow,
    _is_torch,
    _is_torch_device,
    cached_path,
    copy_func,
    get_file_from_repo,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    to_py_obj,
    torch_required,
)
from transformers.utils import logging


if is_tokenizers_available():
    from tokenizers import AddedToken
    from tokenizers import Encoding as EncodingFast
else:

    @dataclass(frozen=True, eq=True)
    class AddedToken:
        content = field(default_factory=str)
        single_word = False
        lstrip = False
        rstrip = False
        normalized = True

        def __getstate__(self):
            return self.__dict__

    @dataclass
    class EncodingFast:
        pass


log = logging.get_logger(__name__)

VERY_LARGE_INTEGER = int(1e30)
LARGE_INTEGER = int(1e20)

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CharSpan(NamedTuple):
    start = None
    end = None


class TokenSpan(NamedTuple):
    start = None
    end = None


class BatchEncoding(UserDict):
    def __init__(
        self,
        data=None,
        encoding=None,
        tensor_type=None,
        prepend_batch_axis=False,
        n_sequences=None,
    ):
        super().__init__(data)
        if isinstance(encoding, EncodingFast):
            encoding = [encoding]
        self._encodings = encoding
        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences
        self._n_sequences = n_sequences
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self):
        return self._n_sequences

    @property
    def is_fast(self):
        return self._encodings is not None

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]
        else:
            return self._encodings[item]

    def __getattr__(self, item):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def encodings(self):
        return self._encodings

    def tokens(self, b=0):
        return self._encodings[b].tokens

    def sequence_ids(self, b=0):
        return self._encodings[b].sequence_ids

    def words(self, b=0):
        return self.word_ids(b)

    def word_ids(self, b=0):
        return self._encodings[b].word_ids

    def token_to_sequence(self, b_or_t, t=None):
        if t is not None:
            b = b_or_t
        else:
            b = 0
            t = b_or_t
        if b < 0:
            b = self._batch_size + b
        if t < 0:
            t = self._seq_len + t
        return self._encodings[b].token_to_sequence(t)

    def token_to_word(self, b_or_t, t=None):
        if t is not None:
            b = b_or_t
        else:
            b = 0
            t = b_or_t
        if b < 0:
            b = self._batch_size + b
        if t < 0:
            t = self._seq_len + t
        return self._encodings[b].token_to_word(t)

    def word_to_tokens(self, b_or_w, w=None, sequence_index=0):
        if w is not None:
            b = b_or_w
        else:
            b = 0
            w = b_or_w
        if b < 0:
            b = self._batch_size + b
        if w < 0:
            w = self._seq_len + w
        span = self._encodings[b].word_to_tokens(w, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self, b_or_t, t=None):
        if t is not None:
            b = b_or_t
        else:
            b = 0
            t = b_or_t
        return CharSpan(*(self._encodings[b].token_to_chars(t)))

    def char_to_token(self, b_or_c, c=None, sequence_index=0):
        if c is not None:
            b = b_or_c
        else:
            b = 0
            c = b_or_c
        return self._encodings[b].char_to_token(c, sequence_index)

    def word_to_chars(self, b_or_w, w=None, sequence_index=0):
        if w is not None:
            b = b_or_w
        else:
            b = 0
            w = b_or_w
        return CharSpan(*(self._encodings[b].word_to_chars(w, sequence_index)))

    def char_to_word(self, b_or_c, c=None, sequence_index=0):
        if c is not None:
            b = b_or_c
        else:
            b = 0
            c = b_or_c
        return self._encodings[b].char_to_word(c, sequence_index)

    def convert_to_tensors(self, tensor_type=None, prepend_batch_axis=False):
        if tensor_type is None:
            return self
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        if tensor_type == TensorType.TENSORFLOW:
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )
        return self

    @torch_required
    def to(self, device):
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            log.warning(
                f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported."
            )
        return self


class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos",
        "eos",
        "unk",
        "sep",
        "pad",
        "cls",
        "msk",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kw):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose

        # We directly set the hidden value to allow initialization with special tokens
        # which are not yet in the vocabulary. Necessary for serialization/de-serialization
        # TODO clean this up at some point (probably by switching to fast tokenizers)
        for key, value in kw.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(
                        f"special token {key} has to be either str or AddedToken but got: {type(value)}"
                    )

    def sanitize_special_tokens(self):
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(self, special_tokens_dict):
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            if self.verbose:
                log.info(f"Assigning {value} to the {key} key of the tokenizer")
            setattr(self, key, value)

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f"Token {value} for key {key} should be a str or an AddedToken instance"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(
        self,
        new_tokens,
        special_tokens=False,
    ):
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens, special_tokens=False):
        raise NotImplementedError

    @property
    def bos(self):
        if self._bos_token is None and self.verbose:
            log.error("Using bos, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos(self):
        if self._eos_token is None and self.verbose:
            log.error("Using eos, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk(self):
        if self._unk_token is None and self.verbose:
            log.error("Using unk, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep(self):
        if self._sep_token is None and self.verbose:
            log.error("Using sep, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad(self):
        if self._pad_token is None and self.verbose:
            log.error("Using pad, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls(self):
        if self._cls_token is None and self.verbose:
            log.error("Using cls, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def msk(self):
        if self._mask_token is None and self.verbose:
            log.error("Using msk, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self):
        if self._additional_special_tokens is None and self.verbose:
            log.error("Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos.setter
    def bos(self, value):
        self._bos_token = value

    @eos.setter
    def eos(self, value):
        self._eos_token = value

    @unk.setter
    def unk(self, value):
        self._unk_token = value

    @sep.setter
    def sep(self, value):
        self._sep_token = value

    @pad.setter
    def pad(self, value):
        self._pad_token = value

    @cls.setter
    def cls(self, value):
        self._cls_token = value

    @msk.setter
    def msk(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def BOS(self):
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos)

    @property
    def EOS(self):
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos)

    @property
    def unk_token_id(self):
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk)

    @property
    def SEP(self):
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep)

    @property
    def PAD(self):
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls)

    @property
    def mask_token_id(self):
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.msk)

    @property
    def additional_special_tokens_ids(self):
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @BOS.setter
    def BOS(self, value):
        self._bos_token = self.convert_tokens_to_ids(value)

    @EOS.setter
    def EOS(self, value):
        self._eos_token = self.convert_tokens_to_ids(value)

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_tokens_to_ids(value)

    @SEP.setter
    def SEP(self, value):
        self._sep_token = self.convert_tokens_to_ids(value)

    @PAD.setter
    def PAD(self, value):
        self._pad_token = self.convert_tokens_to_ids(value)

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_tokens_to_ids(value)

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_tokens_to_ids(value)

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_tokens_to_ids(value) for value in values]

    @property
    def special_tokens_map(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (
                    type(attr_value)(str(attr_value_sub) for attr_value_sub in attr_value)
                    if isinstance(attr_value, (list, tuple))
                    else str(attr_value)
                )
        return set_attr

    @property
    def special_tokens_map_extended(
        self,
    ):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self):
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (
                list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value]
            )
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids


class PreTrainedTokenizerBase(SpecialTokensMixin):
    vocab_fs = {}
    vocab_map = {}
    pretrained_init_configuration = {}
    input_caps = {}
    _auto_class = None
    model_input_names = ["input_ids", "typ_ids", "mask"]
    padding_side = "right"
    truncation_side = "right"
    slow_tokenizer_class = None

    def __init__(self, **kw):
        self.init_inputs = ()
        self.init_kw = copy.deepcopy(kw)
        self.name_or_path = kw.pop("name_or_path", "")
        self._processor_class = kw.pop("processor_class", None)
        model_max_length = kw.pop("model_max_length", kw.pop("max_len", None))
        self.model_max_length = (
            model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        )
        self.padding_side = kw.pop("padding_side", self.padding_side)
        assert self.padding_side in ["right", "left"]
        self.truncation_side = kw.pop("truncation_side", self.truncation_side)
        assert self.truncation_side in ["right", "left"]
        self.model_input_names = kw.pop("model_input_names", self.model_input_names)
        self.deprecation_warnings = {}
        super().__init__(**kw)

    @property
    def max_len_single_sentence(self):
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self):
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value):
        if (
            value == self.model_max_length - self.num_special_tokens_to_add(pair=False)
            and self.verbose
        ):
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                log.warning(
                    "Setting 'max_len_single_sentence' is now deprecated. "
                    "This value is automatically set up."
                )
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. "
                "This value is automatically set up."
            )

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value):
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        if (
            value == self.model_max_length - self.num_special_tokens_to_add(pair=True)
            and self.verbose
        ):
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                log.warning(
                    "Setting 'max_len_sentences_pair' is now deprecated. "
                    "This value is automatically set up."
                )
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            raise ValueError(
                "Setting 'max_len_sentences_pair' is now deprecated. "
                "This value is automatically set up."
            )

    def _set_processor_class(self, processor_class):
        self._processor_class = processor_class

    def __repr__(self):
        return (
            f"{'PreTrainedTokenizerFast' if self.is_fast else 'PreTrainedTokenizer'}(name_or_path='{self.name_or_path}', "
            f"s_vocab={self.s_vocab}, model_max_len={self.model_max_length}, is_fast={self.is_fast}, "
            f"padding_side='{self.padding_side}', truncation_side='{self.truncation_side}', special_tokens={self.special_tokens_map_extended})"
        )

    def get_vocab(self):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kw):
        cache_dir = kw.pop("cache_dir", None)
        force_download = kw.pop("force_download", False)
        resume_download = kw.pop("resume_download", False)
        proxies = kw.pop("proxies", None)
        local_files_only = kw.pop("local_files_only", False)
        use_auth_token = kw.pop("use_auth_token", None)
        revision = kw.pop("revision", None)
        subfolder = kw.pop("subfolder", None)
        from_pipeline = kw.pop("_from_pipeline", None)
        from_auto_class = kw.pop("_from_auto", False)
        user_agent = {
            "file_type": "tokenizer",
            "from_auto_class": from_auto_class,
            "is_fast": "Fast" in cls.__name__,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        if is_offline_mode() and not local_files_only:
            log.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
            pretrained_model_name_or_path
        ):
            if len(cls.vocab_fs) > 1:
                raise ValueError(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not "
                    "supported for this tokenizer. Use a model identifier or the path to a directory instead."
                )
            warnings.warn(
                f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is deprecated and "
                "won't be possible anymore in v5. Use a model identifier or the path to a directory instead.",
                FutureWarning,
            )
            file_id = list(cls.vocab_fs.keys())[0]
            vocab_files[file_id] = pretrained_model_name_or_path
        else:
            # At this point pretrained_model_name_or_path is either a directory or a model identifier name
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            }
            vocab_files_target = {**cls.vocab_fs, **additional_files_names}
            if "tokenizer_file" in vocab_files_target:
                # Try to get the tokenizer config to see if there are versioned tokenizer files.
                fast_tokenizer_file = FULL_TOKENIZER_FILE
                resolved_config_file = get_file_from_repo(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                if resolved_config_file is not None:
                    with open(resolved_config_file, encoding="utf-8") as reader:
                        tokenizer_config = json.load(reader)
                        if "fast_tokenizer_files" in tokenizer_config:
                            fast_tokenizer_file = get_fast_tokenizer_file(
                                tokenizer_config["fast_tokenizer_files"]
                            )
                vocab_files_target["tokenizer_file"] = fast_tokenizer_file
            for file_id, file_name in vocab_files_target.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    if subfolder is not None:
                        full_file_name = os.path.join(
                            pretrained_model_name_or_path, subfolder, file_name
                        )
                    else:
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                    if not os.path.exists(full_file_name):
                        log.info(f"Didn't find file {full_file_name}. We won't load it.")
                        full_file_name = None
                else:
                    full_file_name = hf_bucket_url(
                        pretrained_model_name_or_path,
                        filename=file_name,
                        subfolder=subfolder,
                        revision=revision,
                        mirror=None,
                    )
                vocab_files[file_id] = full_file_name
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            else:
                try:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )

                except FileNotFoundError as error:
                    if local_files_only:
                        unresolved_files.append(file_id)
                    else:
                        raise error

                except RepositoryNotFoundError:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                        "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to "
                        "pass a token having permission to this repo with `use_auth_token` or log in with "
                        "`huggingface-cli login` and pass `use_auth_token=True`."
                    )
                except RevisionNotFoundError:
                    raise EnvironmentError(
                        f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
                        "for this model name. Check the model page at "
                        f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                    )
                except EntryNotFoundError:
                    log.debug(
                        f"{pretrained_model_name_or_path} does not contain a file named {file_path}."
                    )
                    resolved_vocab_files[file_id] = None

                except HTTPError as err:
                    if "404 Client Error" in str(err):
                        log.debug(f"Connection problem to access {file_path}.")
                        resolved_vocab_files[file_id] = None
                    else:
                        raise err

        if len(unresolved_files) > 0:
            log.info(
                f"Can't load following files from cache: {unresolved_files} and cannot check if these "
                "files are necessary for the tokenizer to operate."
            )

        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
            raise EnvironmentError(
                f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing all relevant files for a {cls.__name__} tokenizer."
            )

        for file_id, file_path in vocab_files.items():
            if file_id not in resolved_vocab_files:
                continue

            if file_path == resolved_vocab_files[file_id]:
                log.info(f"loading file {file_path}")
            else:
                log.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")

        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            **kw,
        )

    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        use_auth_token=None,
        cache_dir=None,
        **kw,
    ):
        from_slow = kw.get("from_slow", False)
        has_tokenizer_file = resolved_vocab_files.get("tokenizer_file", None) is not None
        if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None:
            slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
                copy.deepcopy(resolved_vocab_files),
                pretrained_model_name_or_path,
                copy.deepcopy(init_configuration),
                *init_inputs,
                **(copy.deepcopy(kw)),
            )
        else:
            slow_tokenizer = None
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kw = json.load(tokenizer_config_handle)
            config_tokenizer_class = init_kw.get("tokenizer_class")
            init_kw.pop("tokenizer_class", None)
            init_kw.pop("auto_map", None)
            saved_init_inputs = init_kw.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            config_tokenizer_class = None
            init_kw = init_configuration

        if config_tokenizer_class is None:
            from .models.auto.configuration_auto import AutoConfig  # tests_ignore

            try:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                )
                config_tokenizer_class = config.tokenizer_class
            except (OSError, ValueError, KeyError):
                config = None
            if config_tokenizer_class is None:
                from .models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES  # tests_ignore

                if hasattr(config, "model_type"):
                    model_type = config.model_type
                else:
                    # Fallback: use pattern matching on the string.
                    model_type = None
                    for pattern in TOKENIZER_MAPPING_NAMES.keys():
                        if pattern in str(pretrained_model_name_or_path):
                            model_type = pattern
                            break

                if model_type is not None:
                    (
                        config_tokenizer_class,
                        config_tokenizer_class_fast,
                    ) = TOKENIZER_MAPPING_NAMES.get(model_type, (None, None))
                    if config_tokenizer_class is None:
                        config_tokenizer_class = config_tokenizer_class_fast

        if config_tokenizer_class is not None:
            if cls.__name__.replace("Fast", "") != config_tokenizer_class.replace("Fast", ""):
                log.warning(
                    "The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. "
                    "It may result in unexpected tokenization. \n"
                    f"The tokenizer class you load from this checkpoint is '{config_tokenizer_class}'. \n"
                    f"The class this function is called from is '{cls.__name__}'."
                )
        init_kw.update(kw)

        def convert_added_tokens(obj):
            if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
                obj.pop("__type")
                return AddedToken(**obj)
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v) for k, v in obj.items()}
            return obj

        init_kw = convert_added_tokens(init_kw)
        if pretrained_model_name_or_path in cls.input_caps:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            model_max_length = cls.input_caps[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kw["model_max_length"] = min(
                    init_kw.get("model_max_length", int(1e30)), model_max_length
                )

        # Merge resolved_vocab_files arguments in init_kw.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kw:
                init_kw[args_name] = file_path

        if slow_tokenizer is not None:
            init_kw["__slow_tokenizer"] = slow_tokenizer

        init_kw["name_or_path"] = pretrained_model_name_or_path

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kw)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kw for saving and re-loading with ``save_pretrained``
        # Removed: Now done at the base class level
        # tokenizer.init_inputs = init_inputs
        # tokenizer.init_kw = init_kw

        # If there is a complementary special token map, load it
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key in kw and kw[key]:
                    # This value has already been redefined by the kw
                    # We keep this new value and ignore the one stored in the special_tokens_map_file

                    continue

                if isinstance(value, dict):
                    value = AddedToken(**value)
                elif isinstance(value, list):
                    value = [
                        AddedToken(**token) if isinstance(token, dict) else token for token in value
                    ]
                setattr(tokenizer, key, value)

        # Add supplementary tokens.
        special_tokens = tokenizer.all_special_tokens
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)

            # Sort added tokens by index
            added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(), key=lambda x: x[1]))

            for token, index in added_tok_encoder_sorted:
                if (
                    has_tokenizer_file
                    and index != len(tokenizer)
                    and tokenizer.convert_tokens_to_ids(token) != index
                ):
                    # Tokenizer fast: added token needs to either be in the vocabulary with the proper index or the
                    # index is the current length of the tokenizer (not in vocabulary)
                    raise ValueError(
                        f"Wrong index found for {token}: should be {tokenizer.convert_tokens_to_ids(token)} but found "
                        f"{index}."
                    )
                elif not has_tokenizer_file and index != len(tokenizer):
                    # Tokenizer slow: added token cannot already be in the vocabulary so its index needs to be the
                    # current length of the tokenizer.
                    raise ValueError(
                        f"Non-consecutive added token '{token}' found. "
                        f"Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
                    )

                # Safe to call on a tokenizer fast even if token already there.
                tokenizer.add_tokens(token, special_tokens=bool(token in special_tokens))

        # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
        added_tokens = tokenizer.sanitize_special_tokens()
        if added_tokens:
            log.warning_advice(
                "Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained."
            )

        return tokenizer

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kw,
    ):
        if os.path.isfile(save_directory):
            log.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kw.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kw)

        os.makedirs(save_directory, exist_ok=True)

        special_tokens_map_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE,
        )
        tokenizer_config_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE,
        )

        tokenizer_config = copy.deepcopy(self.init_kw)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_fs.keys():
            tokenizer_config.pop(file_id, None)

        # Sanitize AddedTokens
        def convert_added_tokens(obj, add_type_field=True):
            if isinstance(obj, AddedToken):
                out = obj.__getstate__()
                if add_type_field:
                    out["__type"] = "AddedToken"
                return out
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o, add_type_field=add_type_field) for o in obj)
            elif isinstance(obj, dict):
                return {
                    k: convert_added_tokens(v, add_type_field=add_type_field)
                    for k, v in obj.items()
                }
            return obj

        # add_type_field=True to allow dicts in the kw / differentiate from AddedToken serialization
        tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`
        if tokenizer_class.endswith("Fast") and tokenizer_class != "PreTrainedTokenizerFast":
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=tokenizer_config)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))
        log.info(f"tokenizer config file saved in {tokenizer_config_file}")

        # Sanitize AddedTokens in special_tokens_map
        write_dict = convert_added_tokens(self.special_tokens_map_extended, add_type_field=False)
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(write_dict, ensure_ascii=False))
        log.info(f"Special tokens file saved in {special_tokens_map_file}")

        file_names = (tokenizer_config_file, special_tokens_map_file)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            log.info(f"Tokenizer pushed to the hub in this commit: {url}")

        return save_files

    def _save_pretrained(
        self,
        save_directory,
        file_names,
        legacy_format=None,
        filename_prefix=None,
    ):
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, ensure_ascii=False)
                f.write(out_str)
                log.info(f"added tokens file saved in {added_tokens_file}")

        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files + (added_tokens_file,)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        raise NotImplementedError

    def tokenize(self, text, pair=None, add_special_tokens=False, **kw):
        raise NotImplementedError

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_len=None,
        stride=0,
        return_tensors=None,
        **kw,
    ):
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_len=max_len,
            stride=stride,
            return_tensors=return_tensors,
            **kw,
        )

        return encoded_inputs["input_ids"]

    def num_special_tokens_to_add(self, pair=False):
        raise NotImplementedError

    def _get_padding_truncation_strategies(
        self,
        padding=False,
        truncation=False,
        max_len=None,
        pad_to_multiple_of=None,
        verbose=True,
        **kw,
    ):
        old_truncation_strategy = kw.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kw.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_len, it activates truncation for max_len
        if max_len is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    log.warning(
                        "Truncation was not explicitly activated but `max_len` is provided a specific value, "
                        "please use `truncation=True` to explicitly truncate examples to max length. "
                        "Defaulting to 'longest_first' truncation strategy. "
                        "If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy "
                        "more precisely by providing a specific strategy to `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_len'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_len` (e.g. `max_len=45`) or leave max_len to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_len is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_len is not None and (
                        truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_len` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_len'`."
                        )
                    if old_pad_to_max_length is not False:
                        warnings.warn(
                            "Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`."
                        )
                padding_strategy = (
                    PaddingStrategy.LONGEST
                )  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, "
                    "use `truncation=True` to truncate examples to a max length. You can give a specific "
                    "length with `max_len` (e.g. `max_len=45`) or leave max_len to None to truncate to the "
                    "maximal input size of the model (e.g. 512 for Bert). "
                    " If you have pairs of inputs, you can give a specific truncation strategy selected among "
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_len is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_len", False):
                            log.warning(
                                "Asking to pad to max_len but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_len"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_len = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                            "Asking-to-truncate-to-max_len", False
                        ):
                            log.warning(
                                "Asking to truncate to max_len but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_len"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_len = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad or self.PAD < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad` `(tokenizer.pad = tokenizer.eos e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_len is not None
            and (max_len % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_len}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_len, kw

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
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
        def _is_valid_text_input(t):
            if isinstance(t, str):
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                elif isinstance(t[0], str):
                    return True
                elif isinstance(t[0], (list, tuple)):
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
            return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if is_split_into_words:
            is_batched = (
                isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
            )
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_len=max_len,
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
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_len=max_len,
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

    def encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
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
        (
            padding_strategy,
            truncation_strategy,
            max_len,
            kw,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_len=max_len,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kw,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_len=max_len,
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
        raise NotImplementedError

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
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
        (
            padding_strategy,
            truncation_strategy,
            max_len,
            kw,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_len=max_len,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kw,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_len=max_len,
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
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_len=None,
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
                encoded_inputs["mask"] = []
            return encoded_inputs
        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
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

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_len, _ = self._get_padding_truncation_strategies(
            padding=padding, max_len=max_len, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_len=max_len,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_len = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_len=max_len,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        if toks_1 is None:
            return len(toks_0) * [0]
        return [0] * len(toks_0) + [1] * len(toks_1)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return toks_0
        return toks_0 + toks_1

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_len=None,
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
            max_len,
            kw,
        ) = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_len=max_len,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kw,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return typ_ids while setting add_special_tokens to False "
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

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "typ_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = (
            len_ids
            + len_pair_ids
            + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        )

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and max_len
            and total_len > max_len
        ):
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_len,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_len

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            typ_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            typ_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["typ_ids"] = typ_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_len, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_len=max_len,
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

    def truncate_sequences(
        self,
        ids,
        pair_ids=None,
        num_tokens_to_remove=0,
        truncation_strategy="longest_first",
        stride=0,
    ):
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(
                        f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'."
                    )

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg + "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                log.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            log.warning(
                f"Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                f"truncation strategy. So the returned list will always be empty even if some "
                f"tokens have been removed."
            )
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == "right":
                        ids = ids[:-1]
                    elif self.truncation_side == "left":
                        ids = ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:-1]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
            else:
                log.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

    def _pad(
        self,
        encoded_inputs,
        max_len=None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ):
        if return_attention_mask is None:
            return_attention_mask = "mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_len = len(required_input)

        if (
            max_len is not None
            and pad_to_multiple_of is not None
            and (max_len % pad_to_multiple_of != 0)
        ):
            max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_len
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "mask" not in encoded_inputs:
            encoded_inputs["mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_len - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:

                    encoded_inputs["mask"] = encoded_inputs["mask"] + [0] * difference
                if "typ_ids" in encoded_inputs:
                    encoded_inputs["typ_ids"] = (
                        encoded_inputs["typ_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                encoded_inputs[self.model_input_names[0]] = required_input + [self.PAD] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["mask"] = [0] * difference + encoded_inputs["mask"]
                if "typ_ids" in encoded_inputs:
                    encoded_inputs["typ_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["typ_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs[
                        "special_tokens_mask"
                    ]
                encoded_inputs[self.model_input_names[0]] = [self.PAD] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError

    def batch_decode(
        self,
        sequences,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        **kw,
    ):
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kw,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        **kw,
    ):
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kw,
        )

    def _decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        **kw,
    ):
        raise NotImplementedError

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        assert has_specials and toks_1 is None, (
            "You cannot use ``has_specials=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )
        all_special_ids = self.all_special_ids  # cache the property
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in toks_0]
        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string):
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _eventual_warn_about_too_long_sequence(self, ids, max_len, verbose):
        if max_len is None and len(ids) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get(
                "sequence-length-is-longer-than-the-specified-maximum", False
            ):
                log.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    @contextmanager
    def as_target_tokenizer(self):
        yield

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")
        cls._auto_class = auto_class

    def prepare_seq2seq_batch(
        self,
        src_texts,
        tgt_texts=None,
        max_len=None,
        max_target_length=None,
        padding="longest",
        return_tensors=None,
        truncation=True,
        **kw,
    ):
        kw.pop("src_lang", None)
        kw.pop("tgt_lang", None)
        if max_len is None:
            max_len = self.model_max_length
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_len=max_len,
            padding=padding,
            truncation=truncation,
            **kw,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_len
        with self.as_target_tokenizer():
            labels = self(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_len=max_target_length,
                truncation=truncation,
                **kw,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def get_fast_tokenizer_file(tokenization_files):
    tokenizer_files_map = {}
    for file_name in tokenization_files:
        search = _re_tokenizer_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name
    available_versions = sorted(tokenizer_files_map.keys())
    tokenizer_file = FULL_TOKENIZER_FILE
    for v in available_versions:
        tokenizer_file = tokenizer_files_map[v]
    return tokenizer_file


PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)
PreTrainedTokenizerBase.push_to_hub.__doc__ = PreTrainedTokenizerBase.push_to_hub.__doc__.format(
    object="tokenizer", object_class="AutoTokenizer", object_files="tokenizer files"
)
