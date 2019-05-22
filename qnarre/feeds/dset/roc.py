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
import random

import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import features as F
from qnarre.feeds.prep import utils, encoder


def dset(ps, kind):
    t, sh = tf.int32, tf.TensorShape((ps.len_src, ))
    return tf.data.Dataset.from_generator(
        lambda: features(ps, kind),
        ((t, ) * 3, t),
        ((sh, sh, tf.TensorShape(())), tf.TensorShape(())),
    )


def features(ps, kind):
    tokenizer = encoder.tokenizer_for(ps)
    fs = F.Topics(tokenizer(reader(ps, kind)))
    ps.update(features=fs)
    for _, c, q, rep in fs.replies():
        cs, qs = c.toks, q.toks
        if ps.len_qry:
            qs = qs[:ps.len_qry]
        cl, ql = len(cs), len(qs)
        cl = min(cl, ps.len_src - ql - 3)
        src = [ps.CLS] + qs + [ps.SEP] + cs[:cl] + [ps.SEP]
        typ = [0] * ql + [1] * (cl + 1)
        yield (src, typ, rep.uid), 1 if rep.valid else 0


def reader(ps, kind):
    assert not ps.dset or ps.dset == 'roc'
    p = pth.Path(ps.dir_data) / ps.dset
    for n in names[kind]:
        with lzma.open(p / (n + '.csv.xz'), mode='rt') as f:
            for i, ln in enumerate(csv.reader(f)):
                if i < 1:
                    continue
                ln = utils.normalize(ln)
                if kind == 'train':
                    t = ln[1].strip()
                    c = ' '.join(t.strip() for t in ln[2:6])
                    qs = [
                        F.Query(
                            qid=utils.next_uid('query'),
                            text=ln[6].strip(),
                            valid=True,
                            toks=F.Toks(),
                        )
                    ]
                else:
                    t = ''
                    c = ' '.join(t.strip() for t in ln[1:5])
                    tgt = int(ln[-1]) - 1
                    qs = [
                        F.Query(
                            qid=utils.next_uid('query'),
                            text=ln[5].strip(),
                            valid=(tgt == 0),
                            toks=F.Toks(),
                        ),
                        F.Query(
                            qid=utils.next_uid('query'),
                            text=ln[6].strip(),
                            valid=(tgt == 1),
                            toks=F.Toks(),
                        )
                    ]
                c = F.Ctxt(text=c, toks=F.Toks(), queries=qs)
                yield F.Topic(title=t, ctxts=[c])


names = {
    'train': ('rocstories_2016', 'rocstories_2017'),
    'test': ('cloze_val', 'cloze_test'),
}
