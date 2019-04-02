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

from qnarre.feeds.prep.tokenizer import Tokenizer
from qnarre.feeds.prep.layout import (Span, Tokens, Topic, Topics, Context,
                                      Question, Answer)


def dataset(kind, params):
    PS = params
    ts = Topics(Tokenizer(PS)(_reader(kind, PS)))
    return tf.data.Dataset.from_generator(
        lambda: _converter(kind, PS, ts),
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
        (
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([2]),
            tf.TensorShape([1]),
        ),
    )


def _reader(kind, params):
    PS = params
    p = pth.Path(PS.data_dir)
    for n in _names[kind]:
        with lzma.open(p / (n + '.json.xz'), mode='rt') as f:
            for t in json.load(f)['data']:
                cs = []
                for p in t['paragraphs']:
                    ctx = _normalize(p['context'])
                    qs = []
                    for q in p['qas']:
                        ans = []
                        for a in q.get('answers', ()):
                            tx = _normalize(a['text'])
                            s = a['answer_start']
                            if ctx.find(tx, s) == s:
                                ans.append(
                                    Answer(
                                        text=tx,
                                        tokens=Tokens(),
                                        span=Span(s, s + len(tx)),
                                        uid=_next_uid()))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        vs = []
                        for v in q.get('plausible_answers', ()):
                            tx = _normalize(v['text'])
                            s = v['answer_start']
                            if ctx.find(tx, s) == s:
                                vs.append(
                                    Answer(
                                        text=tx,
                                        tokens=Tokens(),
                                        span=Span(s, s + len(tx)),
                                        uid=_next_uid()))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        qs.append(
                            Question(
                                qid=q['id'],
                                text=_normalize(q['question']),
                                unfit=q.get('is_impossible', False),
                                tokens=Tokens(),
                                answers=tuple(ans),
                                viables=tuple(vs)))
                    cs.append(
                        Context(
                            text=ctx, tokens=Tokens(), questions=tuple(qs)))
                yield Topic(title=_normalize(t['title']), contexts=tuple(cs))


def _normalize(txt):
    return ' '.join(unicodedata.normalize('NFD', txt).split())


def _converter(kind, params, topics):
    PS = params
    for _, c, q, ans in topics.answers():
        cs, qs = c.tokens, q.tokens
        if PS.max_qry_len:
            qs = qs[:PS.max_qry_len]
        end, ql = len(cs), len(qs)
        sl = PS.max_seq_len - ql - 3
        ss, b = [], 0
        while b < end:
            e = end
            e = (b + sl) if e - b > sl else e
            ss.append(Span(begin=b, end=e))
            if e == end:
                break
            b = min(e, b + PS.doc_stride)
        ql += 2
        for si, s in enumerate(ss):
            toks = [PS.CLS] + qs + [PS.SEP] + cs[s.begin:s.end] + [PS.SEP]
            segs = [0] * ql + [1] * (len(s) + 1)

            def _optim(i):
                o, oi = None, -1
                for s2i, s2 in enumerate(ss):
                    if i >= s2.begin and i < s2.end:
                        left = i - s2.begin
                        right = s2.end - i - 1
                        o2 = min(left, right) + 0.01 * len(s2)
                        if o is None or o2 > o:
                            o, oi = o2, s2i
                return 1 if si == oi else 0

            optims = [0] * ql
            optims += [_optim(idx) for idx in range(s.begin, s.end)] + [0]
            span = 0, 0
            if kind == 'train':
                if not q.unfit:
                    b, e = ans.span.begin, ans.span.end
                    if b >= s.begin and e <= s.end:
                        b += ql - s.begin
                        e += ql - s.end
                        span = b, e
            yield toks, segs, optims, span, ans.uid


def _next_uid():
    global _uid
    _uid += 1
    return _uid


_uid = 0
_names = {
    'train': ('train-v2.0', 'train-v1.1'),
    'test': ('dev-v2.0', 'dev-v1.1'),
}
