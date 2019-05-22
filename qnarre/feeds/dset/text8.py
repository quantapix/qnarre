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
import zipfile

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

num_test_words = 100000

def reader(ps, kind):
    assert not ps.dset or ps.dset == 'text8'
    p = pth.Path(ps.dir_data) / ps.dset
    with zipfile.ZipFile(p / 'text8.zip') as z:
        with z.open('text8') as f:
            ws = utils.normalize(f.read().decode().strip()).split()
            if kind == 'train':
                ws = ws[:-2 * num_test_words]
            elif kind == 'valid':
                ws = ws[-2 * num_test_words:-num_test_words]
            elif kind == 'test':
                ws = ws[-num_test_words:]

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

def _converter(PS, kind):
    FS = ps.features
    for _, c, q, ans in ps.layout.answers():
        cs, qs = c.tokens, q.tokens
        if ps.max_qry_len:
            qs = qs[:ps.max_qry_len]
        end, ql = len(cs), len(qs)
        sl = ps.max_seq_len - ql - 3
        ss, b = [], 0
        while b < end:
            e = end
            e = (b + sl) if e - b > sl else e
            ss.append(L.Span(begin=b, end=e))
            if e == end:
                break
            b = min(e, b + ps.doc_stride)
        ql += 2
        for si, s in enumerate(ss):
            seq = [ps.CLS] + qs + [ps.SEP] + cs[s.begin:s.end] + [ps.SEP]
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
            pad = [0] * (ps.max_seq_len - len(seq))
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
