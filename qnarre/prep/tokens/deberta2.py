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
import unicodedata

import sentencepiece as sp
import six

from ...tokens.utils import PreTrainedTokenizer


VOCAB_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model",
    }
}

INPUT_CAPS = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}

VOCAB_FS = {"vocab_file": "spm.model"}


class Tokenizer(PreTrainedTokenizer):
    vocab_fs = VOCAB_FS
    vocab_map = VOCAB_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    input_caps = INPUT_CAPS

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        split_by_punct=False,
        bos="[CLS]",
        eos="[SEP]",
        unk="[UNK]",
        sep="[SEP]",
        pad="[PAD]",
        cls="[CLS]",
        msk="[MASK]",
        sp_model_kw=None,
        **kw,
    ):
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        super().__init__(
            do_lower_case=do_lower_case,
            bos=bos,
            eos=eos,
            unk=unk,
            sep=sep,
            pad=pad,
            cls=cls,
            msk=msk,
            split_by_punct=split_by_punct,
            sp_model_kw=self.sp_model_kw,
            **kw,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self._tokenizer = SPMTokenizer(
            vocab_file, split_by_punct=split_by_punct, sp_model_kw=self.sp_model_kw
        )

    @property
    def s_vocab(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        return self._tokenizer.spm.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self._tokenizer.spm.IdToPiece(index) if index < self.s_vocab else self.unk

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, toks_0, toks_1=None):
        if toks_1 is None:
            return [self.cls_token_id] + toks_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + toks_0 + sep + toks_1 + sep

    def get_special_tokens_mask(self, toks_0, toks_1=None, has_specials=False):
        if has_specials:
            return super().get_special_tokens_mask(toks_0=toks_0, toks_1=toks_1, has_specials=True)
        if toks_1 is not None:
            return [1] + ([0] * len(toks_0)) + [1] + ([0] * len(toks_1)) + [1]
        return [1] + ([0] * len(toks_0)) + [1]

    def create_token_type_ids_from_sequences(self, toks_0, toks_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if toks_1 is None:
            return len(cls + toks_0 + sep) * [0]
        return len(cls + toks_0 + sep) * [0] + len(toks_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kw):
        add_prefix_space = kw.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kw)

    def save_vocabulary(self, dir, pre=None):
        return self._tokenizer.save_pretrained(dir, pre=pre)


class SPMTokenizer:
    def __init__(self, vocab_file, split_by_punct=False, sp_model_kw=None):
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.sp_model_kw = {} if sp_model_kw is None else sp_model_kw
        spm = sp.SentencePieceProcessor(**self.sp_model_kw)
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"{vocab_file} does not exist!")
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        # Token map
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        self.spm = spm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kw"):
            self.sp_model_kw = {}
        self.spm = sp.SentencePieceProcessor(**self.sp_model_kw)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        pieces = self._encode_as_pieces(text)

        def _norm(x):
            if x not in self.vocab or x == "<unk>":
                return "[UNK]"
            else:
                return x

        pieces = [_norm(p) for p in pieces]
        return pieces

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            return self.spm.decode_pieces([t for t in tokens])
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        if (
            len(token) == 1
            and (
                _is_whitespace(list(token)[0])
                or _is_control(list(token)[0])
                or _is_punctuation(list(token)[0])
            )
        ) or token in self.special_tokens:
            return False
        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "[MASK]"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode(text, out_type=str)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save_pretrained(self, path, pre=None):
        filename = VOCAB_FS[list(VOCAB_FS.keys())[0]]
        if pre is not None:
            filename = pre + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)


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


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    else:
        raise ValueError("Not running on Python2 or Python 3?")
