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

import collections

from ...tokens.utils import BatchEncoding
from .bert import Bert


VOCAB_FS = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_MAP = {
    "vocab_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/tokenizer.json",
    },
}
QUESTION_ENCODER_PRETRAINED_VOCAB_MAP = {
    "vocab_file": {
        "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/tokenizer.json",
    },
}
READER_PRETRAINED_VOCAB_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/vocab.txt",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/tokenizer.json",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/tokenizer.json",
    },
}

CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,
    "facebook/dpr-ctx_encoder-multiset-base": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,
    "facebook/dpr-question_encoder-multiset-base": 512,
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,
    "facebook/dpr-reader-multiset-base": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-ctx_encoder-multiset-base": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-question_encoder-multiset-base": {"do_lower_case": True},
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},
    "facebook/dpr-reader-multiset-base": {"do_lower_case": True},
}


class DPRContextEncoderTokenizer(Bert):
    vocab_fs = VOCAB_FS
    vocab_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_MAP
    input_caps = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION


class DPRQuestionEncoderTokenizer(Bert):
    vocab_fs = VOCAB_FS
    vocab_map = QUESTION_ENCODER_PRETRAINED_VOCAB_MAP
    input_caps = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION


DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction",
    ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"],
)

DPRReaderOutput = collections.namedtuple(
    "DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"]
)


class Mixin:
    def __call__(
        self,
        questions,
        titles=None,
        texts=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_attention_mask=None,
        **kw,
    ):
        if titles is None and texts is None:
            return super().__call__(
                questions,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kw,
            )
        elif titles is None or texts is None:
            text_pair = titles if texts is None else texts
            return super().__call__(
                questions,
                text_pair,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kw,
            )
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        if len(titles) != len(texts):
            raise ValueError(
                f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
            )
        encoded_question_and_titles = super().__call__(
            questions, titles, padding=False, truncation=False
        )["input_ids"]
        encoded_texts = super().__call__(
            texts, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(
                    encoded_question_and_titles, encoded_texts
                )
            ]
        }
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.PAD) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        return self.pad(
            encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors
        )

    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans=16,
        max_answer_length=64,
        num_spans_per_passage=4,
    ):
        input_ids = reader_input["input_ids"]
        start_logits, end_logits, relevance_logits = reader_output[:3]
        n_passages = len(relevance_logits)
        sorted_docs = sorted(range(n_passages), reverse=True, key=relevance_logits.__getitem__)
        nbest_spans_predictions = []
        for doc_id in sorted_docs:
            sequence_ids = list(input_ids[doc_id])
            # assuming question & title information is at the beginning of the sequence
            passage_offset = sequence_ids.index(self.sep_token_id, 2) + 1  # second sep id
            if sequence_ids[-1] == self.PAD:
                sequence_len = sequence_ids.index(self.PAD)
            else:
                sequence_len = len(sequence_ids)

            best_spans = self._get_best_spans(
                start_logits=start_logits[doc_id][passage_offset:sequence_len],
                end_logits=end_logits[doc_id][passage_offset:sequence_len],
                max_answer_length=max_answer_length,
                top_spans=num_spans_per_passage,
            )
            for start_index, end_index in best_spans:
                start_index += passage_offset
                end_index += passage_offset
                nbest_spans_predictions.append(
                    DPRSpanPrediction(
                        span_score=start_logits[doc_id][start_index]
                        + end_logits[doc_id][end_index],
                        relevance_score=relevance_logits[doc_id],
                        doc_id=doc_id,
                        start_index=start_index,
                        end_index=end_index,
                        text=self.decode(sequence_ids[start_index : end_index + 1]),
                    )
                )
            if len(nbest_spans_predictions) >= num_spans:
                break
        return nbest_spans_predictions[:num_spans]

    def _get_best_spans(
        self,
        start_logits,
        end_logits,
        max_answer_length,
        top_spans,
    ):
        scores = []
        for (start_index, start_score) in enumerate(start_logits):
            for (answer_length, end_score) in enumerate(
                end_logits[start_index : start_index + max_answer_length]
            ):
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        for (start_index, end_index), score in scores:
            if start_index > end_index:
                raise ValueError(f"Wrong span indices: [{start_index}:{end_index}]")
            length = end_index - start_index + 1
            if length > max_answer_length:
                raise ValueError(f"Span is too long: {length} > {max_answer_length}")
            if any(
                [
                    start_index <= prev_start_index <= prev_end_index <= end_index
                    or prev_start_index <= start_index <= end_index <= prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals
                ]
            ):
                continue
            chosen_span_intervals.append((start_index, end_index))

            if len(chosen_span_intervals) == top_spans:
                break
        return chosen_span_intervals


class DPRReaderTokenizer(Mixin, Bert):
    vocab_fs = VOCAB_FS
    vocab_map = READER_PRETRAINED_VOCAB_MAP
    input_caps = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["input_ids", "attention_mask"]
