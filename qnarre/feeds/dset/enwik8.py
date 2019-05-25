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

import zipfile

import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import records as R
from qnarre.feeds.prep import features as F
from qnarre.feeds.prep import utils, encoder


def dset(ps, kind):
    assert ps.dset == 'enwik8'
    p = pth.Path(ps.dir_data) / ps.dset
    pv = p / ps.vocab_path
    p = p / kind
    if not p.exists():
        tokenizer = encoder.tokenizer_for(ps)
        tp = F.Topic(ps.dset, tokenizer(reader(ps, kind)))
        R.dump(p / ps.dset, lambda: recorder(tp))
        if kind == 'train' and not pv.exists():
            R.dump(pv, lambda: [tokenizer.vocab.record()])
    ds = R.dataset(p / ps.dset)
    return ds, feats


feats = {
    'context': tf.VarLenFeature(tf.int64),
    'uid': tf.FixedLenFeature([], tf.string),
}


def recorder(topic):
    for _, c in topic.contexts():
        yield R.example({
            'context': R.ints_feat([*c.toks]),
            'uid': R.bytes_feat(c.uid),
        })


def reader(ps, kind):
    assert not ps.dset or ps.dset == 'enwik8'
    p = pth.Path(ps.dir_data) / ps.dset
    with zipfile.ZipFile(p / 'enwik8.zip') as z:
        with z.open('enwik8') as f:
            ws = utils.normalize(f.read().decode().strip()).split()
            split = ps.test_train_split or 10
            n = len(ws) * split // 100
            if kind == 'train':
                ws = ws[:-2 * n]
            elif kind == 'valid':
                ws = ws[-2 * n:-n]
            elif kind == 'test':
                ws = ws[-n:]
            wl = ps.len_words
            for i in range(len(ws) // wl):
                cu = '{:0>9d}0'.format(i)
                yield F.Context(ws[i * wl:(i + 1) * wl], uid=cu)
