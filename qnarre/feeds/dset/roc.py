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

import csv
import lzma

import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import records as R
from qnarre.feeds.prep import features as F
from qnarre.feeds.prep import utils, encoder


def dset(ps, kind):
    assert ps.dset == 'roc'
    p = pth.Path(ps.dir_data) / ps.dset
    pv = p / ps.vocab_path
    p = p / kind
    if not p.exists():
        tokenizer = encoder.tokenizer_for(ps)
        ts = F.Topics(tokenizer(reader(ps, kind)))
        R.dump(p / ps.dset, lambda: recorder(ts))
        if kind == 'train' and not pv.exists():
            R.dump(pv, lambda: [tokenizer.vocab.record()])
    ds = R.dataset(p / ps.dset)
    return ds, feats


feats = {
    'title': tf.VarLenFeature(tf.int64),
    'context': tf.VarLenFeature(tf.int64),
    'query': tf.VarLenFeature(tf.int64),
    'valid': tf.FixedLenFeature([], tf.int64),
    'uid': tf.FixedLenFeature([], tf.string),
},


def recorder(topics):
    for t, c, q in topics.queries():
        yield R.example({
            'title': R.ints_feat([*t.title.toks]),
            'context': R.ints_feat([*c.toks]),
            'query': R.ints_feat([*q.toks]),
            'valid': R.one_int_feat(1 if q.valid else 0),
            'uid': R.bytes_feat(q.uid),
        })


def reader(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    for n in registry[kind]:
        with lzma.open(p / (n + '.csv.xz'), mode='rt') as f:
            for i, ln in enumerate(csv.reader(f)):
                if i:
                    ln = utils.normalize(ln)
                    if kind == 'train':
                        tt = ln[1].strip()
                        ct = ' '.join(t.strip() for t in ln[2:6])
                        qs = [F.Query(ln[6].strip(), True, ln[0].strip())]
                    else:
                        tt = ''
                        ct = ' '.join(t.strip() for t in ln[1:5])
                        qu = ln[0].strip()
                        v = int(ln[-1])
                        qs = [
                            F.Query(ln[5].strip(), v == 1, qu + f'-r0'),
                            F.Query(ln[6].strip(), v == 2, qu + f'-r1'),
                        ]
                    yield F.Topic(tt, [F.Context(ct, qs)])


registry = {
    'train': ('rocstories_2016', 'rocstories_2017'),
    'test': ('cloze_val', 'cloze_test'),
}
