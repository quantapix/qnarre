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

import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import features as F
from qnarre.feeds.prep import utils, encoder


def dset(ps, kind):
    t, sh = tf.int32, tf.TensorShape((ps.len_src, ))
    return tf.Dataset.from_generator(
        lambda: features(ps, kind),
        ((t, ) * 4, (t, ) * 2),
        ((sh, ) * 4, tf.TensorShape(())),
    )


def features(ps, kind):
    tokenizer = encoder.tokenizer_for(ps)
    fs = F.Topics(tokenizer(reader(ps, kind)))
    ps.update(features=fs)
    for _, c, q, rep in fs.replies():
        cs, qs = c.toks, q.toks
        if ps.len_qry:
            qs = qs[:ps.len_qry]
        end, ql = len(cs), len(qs)
        sl = ps.len_src - ql - 3
        ss, b = [], 0
        while b < end:
            e = end
            e = (b + sl) if e - b > sl else e
            ss.append(F.Span(begin=b, end=e))
            if e == end:
                break
            b = min(e, b + ps.src_stride)
        ql += 2
        for si, s in enumerate(ss):
            src = [ps.CLS] + qs + [ps.SEP] + cs[s.begin:s.end] + [ps.SEP]
            typ = [0] * ql + [1] * (len(s) + 1)

            def optim(i):
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
            opt += [optim(idx) for idx in range(s.begin, s.end)] + [0]
            assert len(src) == len(typ) == len(opt)
            pad = [0] * (ps.len_src - len(src))
            if pad:
                src += pad
                typ += pad
                opt += pad
            beg, end = 0, 0
            if kind == 'train':
                if not q.valid:
                    beg, end = rep.span.begin, rep.span.end
                    if b >= s.begin and e <= s.end:
                        beg += ql - s.begin
                        end += ql - s.end
            yield (src, typ, opt, rep.uid), (beg, end)


def reader(ps, kind):
    assert not ps.dset or ps.dset == 'squad'
    p = pth.Path(ps.data_dir) / ps.dset
    for n in names[kind]:
        with lzma.open(p / (n + '.json.xz'), mode='rt') as f:
            for t in json.load(f)['data']:
                cs = []
                for p in t['paragraphs']:
                    ctx = utils.normalize(p['context'])
                    qs = []
                    for q in p['qas']:
                        rs = []
                        for r in q.get('answers', ()):
                            tx = utils.normalize(r['text'])
                            s = r['answer_start']
                            if ctx.find(tx, s) == s:
                                rs.append(
                                    F.Reply(
                                        text=tx,
                                        toks=F.Toks(),
                                        span=F.Span(s, s + len(tx)),
                                        uid=utils.next_uid('reply'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        ps = []
                        for p in q.get('plausible_answers', ()):
                            tx = utils.normalize(p['text'])
                            s = p['answer_start']
                            if ctx.find(tx, s) == s:
                                ps.append(
                                    F.Reply(
                                        text=tx,
                                        toks=F.Toks(),
                                        span=F.Span(s, s + len(tx)),
                                        uid=utils.next_uid('reply'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        qs.append(
                            F.Query(
                                qid=q['id'],
                                text=utils.normalize(q['question']),
                                valid=q.get('is_impossible', False),
                                toks=F.Toks(),
                                replies=rs,
                                plaus=ps,
                            ))
                    cs.append(F.Ctxt(text=ctx, toks=F.Toks(), queries=qs))
                yield F.Topic(title=utils.normalize(t['title']), ctxts=cs)


names = {
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
