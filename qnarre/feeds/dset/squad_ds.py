# Copyright 2019 Quantapix Authors. All Rights Reserved.
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
import lzma
import unicodedata

import pathlib as pth

import tensorflow as tf

from qfeeds.prep.tokenizer import Tokenizer
from qfeeds.prep.layout import (Span, Tokens, Topic, Topics, Context, Question,
                                Answer)

_names = {
    'train': ('train-v2.0', 'train-v1.1'),
    'test': ('dev-v2.0', 'dev-v1.1'),
}


def normalize(txt):
    return ' '.join(unicodedata.normalize('NFD', txt).split())


def dataset(kind, params):
    PS = params

    def _reader():
        p = pth.Path(PS.data_dir)
        for n in _names[kind]:
            with lzma.open(p / (n + '.json.xz'), mode='rt') as f:
                for t in json.load(f)['data']:
                    cs = []
                    for p in t['paragraphs']:
                        ctx = normalize(p['context'])
                        qs = []
                        for q in p['qas']:
                            ans = []
                            for a in q.get('answers', ()):
                                tx = normalize(a['text'])
                                s = a['answer_start']
                                if ctx.find(tx, s) == s:
                                    ans.append(
                                        Answer(
                                            text=tx,
                                            span=Span(s, s + len(tx)),
                                            tokens=Tokens()))
                                else:
                                    print('Mismatched', ctx[:20], tx[:20])
                            vs = []
                            for v in q.get('plausible_answers', ()):
                                tx = normalize(v['text'])
                                s = v['answer_start']
                                if ctx.find(tx, s) == s:
                                    vs.append(
                                        Answer(
                                            text=tx,
                                            span=Span(s, s + len(tx)),
                                            tokens=Tokens()))
                                else:
                                    print('Mismatched', ctx[:20], tx[:20])
                            qs.append(
                                Question(
                                    qid=q['id'],
                                    text=normalize(q['question']),
                                    unfit=q.get('is_impossible', False),
                                    tokens=Tokens(),
                                    answers=tuple(ans),
                                    viables=tuple(vs)))
                        cs.append(
                            Context(
                                text=ctx, tokens=Tokens(),
                                questions=tuple(qs)))
                    yield Topic(
                        title=normalize(t['title']), contexts=tuple(cs))

    tokenizer = Tokenizer(PS)
    topics = Topics(tokenizer(_reader()))
    CLS, SEP, MASK = layout.CLS, layout.SEP, layout.MASK

    def _converter(topics, training):

        unique_id = 1000000000
        for _, ctx, qst, ans in topics.answers():
            qts = qst.tokens
            if len(qts) > PS.max_qry_len:
                qts = qts[:PS.max_qry_len]
            beg, end = None, None
            if training:
                if qst.unfit:
                    beg, end = -1, -1
                else:
                    beg, end = ans.span.begin, ans.span.end
                    end = min(end, len(ctx.tokens) - 1)
                    beg, end = _improve_answer_span(ctx.tokens, beg, end,
                                                    tokenizer, ans.text)
            seq_len = PS.max_seq_len - len(qts) - 3
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            off = 0
            while off < len(ctx.tokens):
                length = len(ctx.tokens) - off
                if length > seq_len:
                    length = seq_len
                doc_spans.append(_DocSpan(start=off, length=length))
                if off + length == len(ctx.tokens):
                    break
                off += min(length, PS.doc_stride)

            for doc_span_index, doc_span in enumerate(doc_spans):
                toks = [CLS] + qts + [SEP]
                segs = [0] * (len(qts) + 2) + [1] * (doc_span.length + 1)
                token_to_orig_map = {}
                token_is_max_context = {}
                for i in range(doc_span.length):
                    i = doc_span.start + i
                    token_to_orig_map[len(toks)] = tok_to_orig_index[i]

                    is_max_context = _check_is_max_context(
                        doc_spans, doc_span_index, i)
                    token_is_max_context[len(toks)] = is_max_context
                    toks.append(ctx.tokens[i])
                toks.append(SEP)
                assert len(toks) == PS.max_seq_len
                assert len(segs) == PS.max_seq_len

                start_position = None
                end_position = None
                if training and not qst.unfit:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (beg >= doc_start and end <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(qts) + 2
                        start_position = beg - doc_start + doc_offset
                        end_position = end - doc_start + doc_offset

                if training and qst.unfit:
                    start_position = 0
                    end_position = 0

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    toks=toks,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=qst.unfit)

                # Run callback
                output_fn(feature)

                unique_id += 1
        """
            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)

        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

         """

    return tf.data.Dataset.from_generator(
        _converter,
        (tf.int32, tf.int32, tf.int32),
        (
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([1]),
        ),
    )


PS = None

# ==========================================================


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def input_fn_builder(input_file,
                     seq_length,
                     training,
                     drop_remainder,
                     hvd=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""

        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if training:
            d = tf.data.TFRecordDataset(input_file, num_parallel_reads=4)
            if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
            d = d.apply(tf.data.experimental.ignore_errors())
            d = d.shuffle(buffer_size=100)
            d = d.repeat()
        else:
            d = tf.data.TFRecordDataset(input_file)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
