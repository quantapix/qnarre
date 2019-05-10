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

import random
import collections

import tensorflow as tf

MLM = collections.namedtuple('MLM', ('idx', 'val'))


def dataset(params, data):
    PS = params
    ds = tf.data.Dataset.from_generator(
        lambda: _converter(PS, data),
        (
            tf.int32,  # seq
            tf.int32,  # typ
            tf.int32,  # mlm_idx
            tf.int32,  # mlm_val
            tf.bool,  # fit
        ),
        (
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
        ),
    )
    return ds, data


def _converter(PS, data):
    rng = random.Random(PS.random_seed)
    max_seq_len = PS.max_seq_len - 3
    seq_len = max_seq_len
    if rng.random() < PS.short_seq_prob:
        seq_len = rng.randint(2, max_seq_len)

    def _expand(sgA, sgB, fit):
        sgA = [t for s in sgA for ts in s.tokens for t in ts]
        sgB = [t for s in sgB for ts in s.tokens for t in ts]
        al, bl = len(sgA), len(sgB)
        assert al and bl
        while al + bl > seq_len:
            r = rng.random()
            if al > bl:
                sgA.pop(0 if r < 0.5 else -1)
                al -= 1
            else:
                sgB.pop(0 if r < 0.5 else -1)
                bl -= 1
        seq = [PS.CLS] + sgA + [PS.SEP] + sgB + [PS.SEP]
        typ = [0] * (len(sgA) + 2) + [1] * (len(sgB) + 1)
        seq, idx, val = _mlm_mask(seq)
        yield seq, typ, idx, val, fit

    def _mlm_mask(toks):
        idx = []
        for i, t in enumerate(toks):
            if t != PS.CLS and t != PS.SEP:
                idx.append(i)
        rng.shuffle(idx)
        n = min(PS.mlm_preds, max(1, int(round(len(toks) * PS.mlm_prob))))
        seq = list(toks)
        ms, used = [], set()
        for i in idx:
            if i not in used:
                used.add(i)
                if rng.random() < 0.8:
                    seq[i] = PS.MASK
                elif rng.random() < 0.5:
                    seq[i] = PS.vocab[rng.randint(0, len(PS.vocab) - 1)]
                ms.append(MLM(idx=i, val=toks[i]))
                if len(ms) >= n:
                    break
        idx, val = [], []
        for m in sorted(ms, key=lambda x: x.idx):
            idx.append(m.idx)
            val.append(m.val)
        return seq, idx, val

    for d in data.docs:
        rs = d.rands
        for p in d.paras:
            if p.fit or p.unfit:
                al = sum(len(s.tokens) for s in p.sents)
                bl = len((p.fit or p.unfit).tokens)
                if al + bl <= seq_len:
                    fit = True if p.fit else False
                    yield from _expand(p.sents, [p.fit or p.unfit], fit)
            else:
                sgA = []
                for s in p.sents:
                    if sum(len(s.tokens) for s in sgA) + len(s) <= seq_len:
                        sgA.append(s)
                    elif sgA:
                        sgB, save, fit = [], [], None
                        if len(sgA) > 1:
                            e = rng.randint(1, len(sgA) - 1)
                            save = sgA[e:]
                            sgA = sgA[:e]
                            al = sum(len(s.tokens) for s in sgA)
                        if rs and (len(sgA) == 1 or rng.random() < 0.5):
                            while rs:
                                i = rng.randrange(len(rs))
                                bl = sum(len(s.tokens) for s in sgB)
                                if al + bl + len(rs[i].tokens) <= seq_len:
                                    sgB.append(rs.pop(i))
                                else:
                                    break
                        if not sgB:
                            sgB, fit = save, True
                            save = []
                        if sgB:
                            yield from _expand(sgA, sgB, fit)
                            sgA = save
                        sgA.append(s)
                while len(sgA) > 1:
                    save = []
                    while sum(len(s.tokens) for s in sgA) > seq_len:
                        save.append(sgA.pop(-1))
                    yield from _expand(sgA[:-1], [sgA[-1]], True)
