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
from qnarre.feeds.prep import records as R
from qnarre.feeds.prep import features as F
from qnarre.feeds.prep import utils, encoder


def dset(ps, kind):
    assert ps.dset == 'squad'
    p = pth.Path(ps.dir_data) / ps.dset
    pv = p / ps.vocab_path
    p = p / kind
    if not p.exists():
        tokenizer = encoder.tokenizer_for(ps)
        ts = F.Topics(tokenizer(reader(ps, kind)))
        for n in registry['all']:
            R.dump(p / n, lambda: registry[n](ts))
        if kind == 'train' and not pv.exists():
            R.dump(pv, lambda: [tokenizer.vocab.record()])
    ds = R.dataset(p / ps.dset_subset)
    return ds, feats[ps.dset_subset]


feats = {
    'query_valid': {
        'title': tf.VarLenFeature(tf.int64),
        'context': tf.VarLenFeature(tf.int64),
        'query': tf.VarLenFeature(tf.int64),
        'valid': tf.FixedLenFeature([], tf.int64),
        'uid': tf.FixedLenFeature([], tf.string),
    },
    'reply_spans': {
        'title': tf.VarLenFeature(tf.int64),
        'context': tf.VarLenFeature(tf.int64),
        'query': tf.VarLenFeature(tf.int64),
        'reply': tf.VarLenFeature(tf.int64),
        'begin': tf.FixedLenFeature([], tf.int64),
        'end': tf.FixedLenFeature([], tf.int64),
        'uid': tf.FixedLenFeature([], tf.string),
    },
    'possibles': {
        'title': tf.VarLenFeature(tf.int64),
        'context': tf.VarLenFeature(tf.int64),
        'query': tf.VarLenFeature(tf.int64),
        'possib': tf.VarLenFeature(tf.int64),
        'begin': tf.FixedLenFeature([], tf.int64),
        'end': tf.FixedLenFeature([], tf.int64),
        'uid': tf.FixedLenFeature([], tf.string),
    },
}


def query_valid(topics):
    for t, c, q in topics.queries():
        yield R.example({
            'title': R.ints_feat([*t.title.toks]),
            'context': R.ints_feat([*c.toks]),
            'query': R.ints_feat([*q.toks]),
            'valid': R.one_int_feat(1 if q.valid else 0),
            'uid': R.bytes_feat(q.uid),
        })


def reply_spans(topics):
    for t, c, q, r in topics.replies():
        yield R.example({
            'title': R.ints_feat([*t.title.toks]),
            'context': R.ints_feat([*c.toks]),
            'query': R.ints_feat([*q.toks]),
            'reply': R.ints_feat([*r.toks]),
            'begin': R.one_int_feat(r.span.beg),
            'end': R.one_int_feat(r.span.end),
            'uid': R.bytes_feat(r.uid),
        })


def possibles(topics):
    for t, c, q, p in topics.possibs():
        yield R.example({
            'title': R.ints_feat([*t.title.toks]),
            'context': R.ints_feat([*c.toks]),
            'query': R.ints_feat([*q.toks]),
            'possib': R.ints_feat([*p.toks]),
            'begin': R.one_int_feat(p.span.beg),
            'end': R.one_int_feat(p.span.end),
            'uid': R.bytes_feat(p.uid),
        })


def reader(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    for n in registry[kind]:
        with lzma.open(p / (n + '.json.xz'), mode='rt') as f:
            for data in json.load(f)['data']:
                cs = []
                for p in data['paragraphs']:
                    ct = utils.normalize(p['context'])
                    qs = []
                    for q in p['qas']:
                        qu = q['id']
                        rs = []
                        for i, r in enumerate(q.get('answers', ())):
                            rt = utils.normalize(r['text'])
                            s = r['answer_start']
                            if ct.find(rt, s) == s:
                                s = F.Span(s, s + len(rt))
                                rs.append(F.Reply(rt, s, qu + f'-r{i}'))
                            else:
                                print('Mismatched', ct[:20], rt[:20])
                        ps = []
                        for i, p in enumerate(q.get('plausible_answers', ())):
                            pt = utils.normalize(p['text'])
                            s = p['answer_start']
                            if ct.find(pt, s) == s:
                                s = F.Span(s, s + len(pt))
                                ps.append(F.Reply(pt, s, qu + f'-p{i}'))
                            else:
                                print('Mismatched', ct[:20], pt[:20])
                        qt = utils.normalize(q['question'])
                        qv = q.get('is_impossible', False)
                        qs.append(F.Query(qt, qv, qu, rs, ps))
                    cs.append(F.Context(ct, qs))
                tt = utils.normalize(data['title'])
                yield F.Topic(tt, cs)


registry = {
    'all': ('query_valid', 'reply_spans', 'possibles'),
    'possibles': possibles,
    'query_valid': query_valid,
    'reply_spans': reply_spans,
    'test': ('dev-v2.0', 'dev-v1.1'),
    'train': ('train-v2.0', 'train-v1.1'),
}


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
