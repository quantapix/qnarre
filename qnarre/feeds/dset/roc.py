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
import tensorflow as tf

from qfeeds.prep.encoder import Encoder


def dataset(kind, params):
    PS = params
    enc = Encoder.load(PS)
    msl = mtl = 0

    def _reader():
        p = pth.Path(PS.data_dir)
        for n in _names[kind]:
            with lzma.open(p / (n + '.csv.xz'), mode='rt') as f:
                for i, ln in enumerate(csv.reader(f)):
                    if i > 0:
                        if kind == 'train':
                            t = ln[1]
                            src = ln[2:6]
                            y = random.randint(0, 1)
                            tgt = [
                                ln[6] if y == 0 else '',
                                ln[6] if y == 1 else ''
                            ]
                        else:
                            t = ''
                            src = ln[1:5]
                            tgt = [ln[5], ln[6]]
                            y = int(ln[-1]) - 1
                        yield t, src, tgt, y
                    # if i > 5:
                    #     break

    def _converter():
        nonlocal msl, mtl
        for _, src, tgt, y in _reader():
            tgt = [enc.encode(t) if t else [] for t in tgt]
            tl = max(tgt, key=lambda x: len(x))
            mtl = max(tl, mtl)
            if not PS.tgt_len or tl <= PS.tgt_len:
                ss = [enc.encode(s) if s else [] for s in src]
                ssl = sum(ss, key=lambda x: len(x))
                msl = max(ssl, msl)
                ml = PS.mem_len - tl
                src, ssl = [], 0
                for s in ss:
                    sl = len(s)
                    ml -= sl
                    if ml < 0:
                        break
                    ssl += sl
                    if PS.src_len and ssl > PS.src_len:
                        break
                    src.extend(s)
                if src:
                    for i, t in enumerate(tgt):
                        if t:
                            yield src, t, int(i == y)

    if msl or mtl:
        PS.ds_src_len, PS.ds_tgt_len = msl, mtl
    return tf.data.Dataset.from_generator(
        _converter,
        (tf.int32, tf.int32, tf.int32),
        (
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([1]),
        ),
    )


_names = {
    'train': ('rocstories_2016', 'rocstories_2017'),
    'test': ('cloze_val', 'cloze_test'),
}
