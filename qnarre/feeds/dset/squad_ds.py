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

import pathlib as P
import tensorflow as T

from qnarre.feeds.prep import utils as U
from qnarre.feeds.prep import layout as L


def dataset(params, kind):
    PS = params
    PS.update(layout=L.Topics(PS.tokenizer(_reader(PS, kind))))
    return T.data.Dataset.from_generator(
        lambda: _converter(PS, kind),
        PS.features.tf_dtypes,
        PS.features.tf_shapes,
    )


def _reader(PS, kind):
    p = P.Path(PS.data_dir)
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
                                    L.Answer(
                                        text=tx,
                                        tokens=L.Tokens(),
                                        span=L.Span(s, s + len(tx)),
                                        uid=U.next_uid('answer'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        vs = []
                        for v in q.get('plausible_answers', ()):
                            tx = _normalize(v['text'])
                            s = v['answer_start']
                            if ctx.find(tx, s) == s:
                                vs.append(
                                    L.Answer(
                                        text=tx,
                                        tokens=L.Tokens(),
                                        span=L.Span(s, s + len(tx)),
                                        uid=U.next_uid('answer'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        qs.append(
                            L.Question(
                                qid=q['id'],
                                text=_normalize(q['question']),
                                unfit=q.get('is_impossible', False),
                                tokens=L.Tokens(),
                                answers=ans,
                                viables=vs,
                            ))
                    cs.append(
                        L.Context(
                            text=ctx,
                            tokens=L.Tokens(),
                            questions=qs,
                        ))
                yield L.Topic(
                    title=_normalize(t['title']),
                    contexts=cs,
                )


def _normalize(txt):
    return ' '.join(unicodedata.normalize('NFD', txt).split())


def _converter(PS, kind):
    FS = PS.features
    for _, c, q, ans in PS.layout.answers():
        cs, qs = c.tokens, q.tokens
        if PS.max_qry_len:
            qs = qs[:PS.max_qry_len]
        end, ql = len(cs), len(qs)
        sl = PS.max_seq_len - ql - 3
        ss, b = [], 0
        while b < end:
            e = end
            e = (b + sl) if e - b > sl else e
            ss.append(L.Span(begin=b, end=e))
            if e == end:
                break
            b = min(e, b + PS.doc_stride)
        ql += 2
        for si, s in enumerate(ss):
            seq = [PS.CLS] + qs + [PS.SEP] + cs[s.begin:s.end] + [PS.SEP]
            typ = [0] * ql + [1] * (len(s) + 1)

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

            opt = [0] * ql
            opt += [_optim(idx) for idx in range(s.begin, s.end)] + [0]
            assert len(seq) == len(typ) == len(opt)
            pad = [0] * (PS.max_seq_len - len(seq))
            if pad:
                seq += pad
                typ += pad
                opt += pad
            beg, end = 0, 0
            if kind == 'train':
                if not q.unfit:
                    beg, end = ans.span.begin, ans.span.end
                    if b >= s.begin and e <= s.end:
                        beg += ql - s.begin
                        end += ql - s.end
            yield seq, typ, opt, beg, end, ans.uid


_names = {
    'train': ('train-v2.0', 'train-v1.1'),
    'test': ('dev-v2.0', 'dev-v1.1'),
}
"""
class FeatureWriter:
    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = T.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        self.num_features += 1

        def create_int_feature(values):
            feature = T.train.Feature(
                int64_list=T.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature(
                [feature.start_position])
            features["end_positions"] = create_int_feature(
                [feature.end_position])
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = T.train.Example(
            features=T.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()
"""
