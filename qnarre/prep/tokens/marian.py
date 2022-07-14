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
import os
import re
import warnings
import sentencepiece

from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile

from ...tokens.utils import PreTrainedTokenizer

VOCAB_FS = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

VOCAB_MAP = {
    "source_spm": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/source.spm"
    },
    "target_spm": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/target.spm"
    },
    "vocab": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json"
    },
    "tokenizer_config_file": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/tokenizer_config.json"
    },
}

INPUT_CAPS = {"Helsinki-NLP/opus-mt-en-de": 512}
PRETRAINED_INIT_CONFIGURATION = {}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS
    model_input_names = ["input_ids", "mask"]
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    def __init__(
        self,
        vocab,
        source_spm,
        target_spm,
        source_lang=None,
        target_lang=None,
        unk="<unk>",
        eos="</s>",
        pad="<pad>",
        model_max_length=512,
        sp_model_kw=None,
        **kw,
    ):
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            source_lang=source_lang,
            target_lang=target_lang,
            unk=unk,
            eos=eos,
            pad=pad,
            model_max_length=model_max_length,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"
        self.encoder = load_json(vocab)
        if self.unk not in self.encoder:
            raise KeyError("<unk> token must be in vocab")
        assert self.pad in self.encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.supported_language_codes: list = [
            k for k in self.encoder if k.startswith(">>") and k.endswith("<<")
        ]
        self.spm_files = [source_spm, target_spm]
        self.spm_source = load_spm(source_spm, self.sp_model_kw)
        self.spm_target = load_spm(target_spm, self.sp_model_kw)
        self.current_spm = self.spm_source
        self._setup_normalizer()

    def _setup_normalizer(self):
        try:
            from sacremoses import MosesPunctNormalizer

            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x

    def normalize(self, x):
        return self.punc_normalizer(x) if x else ""

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk])

    def remove_language_code(self, text):
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    def _tokenize(self, text):
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.encode(text, out_type=str)
        return code + pieces

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk)

    def batch_decode(self, sequences, **kw):
        return super().batch_decode(sequences, **kw)

    def decode(self, token_ids, **kw):
        return super().decode(token_ids, **kw)

    def convert_tokens_to_string(self, tokens):
        if self._decode_use_source_tokenizer:
            return self.spm_source.DecodePieces(tokens)
        else:
            return self.spm_target.DecodePieces(tokens)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return toks_0 + [self.EOS]
        return toks_0 + toks_1 + [self.EOS]

    @contextmanager
    def as_target_tokenizer(self):
        self.current_spm = self.spm_target
        yield
        self.current_spm = self.spm_source

    @property
    def s_vocab(self):
        return len(self.encoder)

    def save_vocabulary(self, dir, pre=None):
        saved_files = []
        path = os.path.join(
            dir,
            (pre + "-" if pre else "") + VOCAB_FS["vocab"],
        )
        save_json(self.encoder, path)
        saved_files.append(path)
        for spm_save_filename, spm_orig_path, spm_model in zip(
            [VOCAB_FS["source_spm"], VOCAB_FS["target_spm"]],
            self.spm_files,
            [self.spm_source, self.spm_target],
        ):
            spm_save_path = os.path.join(
                dir,
                (pre + "-" if pre else "") + spm_save_filename,
            )
            if os.path.abspath(spm_orig_path) != os.path.abspath(spm_save_path) and os.path.isfile(
                spm_orig_path
            ):
                copyfile(spm_orig_path, spm_save_path)
                saved_files.append(spm_save_path)
            elif not os.path.isfile(spm_orig_path):
                with open(spm_save_path, "wb") as fi:
                    content_spiece_model = spm_model.serialized_model_proto()
                    fi.write(content_spiece_model)
                saved_files.append(spm_save_path)

        return tuple(saved_files)

    def get_vocab(self):
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(
            {k: None for k in ["spm_source", "spm_target", "current_spm", "punc_normalizer"]}
        )
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kw"):
            self.sp_model_kw = {}
        self.spm_source, self.spm_target = (load_spm(f, self.sp_model_kw) for f in self.spm_files)
        self.current_spm = self.spm_source
        self._setup_normalizer()

    def num_special_tokens_to_add(self, *args, **kw):
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)
        all_special_ids.remove(self.unk_token_id)
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self,
        toks_0,
        toks_1=None,
        has_specials=False,
    ):
        if has_specials:
            return self._special_token_mask(toks_0)
        elif toks_1 is None:
            return self._special_token_mask(toks_0) + [1]
        else:
            return self._special_token_mask(toks_0 + toks_1) + [1]


def load_spm(path, sp_model_kw):
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kw)
    spm.Load(path)
    return spm


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
