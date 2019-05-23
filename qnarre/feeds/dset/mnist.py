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

import gzip

import numpy as np
import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import records as R


def dset(ps, kind):
    assert ps.dset.startswith('mnist')
    p = pth.Path(ps.dir_data) / ps.dset / kind
    if not p.exists():
        ds = tuple(reader(ps, kind))
        print(len(ds))
        R.dump(p / ps.dset, lambda: examples(data=ds))
    ds = tf.TFRecordDataset(str(p / ps.dset))
    ds = ds.map(lambda x: tf.parse_single_example(x, specs()))
    return ds


def specs():
    return {
        'image': tf.VarLenFeature(tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    }


def examples(data):
    for i, l in data:
        f = {
            'image': R.floats_feat(i),
            'label': R.one_int_feat(l),
        }
        e = tf.Example(features=tf.Features(feature=f))
        yield e.SerializeToString()


def reader(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    x, y = registry[kind]
    with gzip.open(p / (x + '.gz'), mode='rb') as xf:
        assert read32(xf) == 2051
        _, r, c = read32(xf), read32(xf), read32(xf)
        with gzip.open(p / (y + '.gz'), mode='rb') as yf:
            assert read32(yf) == 2049
            _ = read32(yf)
            while True:
                x, y = read32(xf, r * c * 4), read32(yf, 1)
                if x is None or y is None:
                    break
                yield x / 255.0, int(y)


def read32(f, count=4):
    b = f.read(count)
    if b:
        dt = np.uint8 if count == 1 else np.dtype(np.uint32).newbyteorder('>')
        rs = np.frombuffer(b, dtype=dt)
        if count <= 4:
            return rs[0]
        return np.array(rs, dtype=np.float)


registry = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'test': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    'try': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
}
