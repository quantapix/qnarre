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

import lzma

import numpy as np
import pathlib as pth

from qnarre.neura import tf


def dset(ps, kind):
    w, h = ps.img_width, ps.img_height
    return tf.Dataset.from_generator(
        lambda: reader(ps, kind),
        (tf.float32, tf.int32),
        (tf.TensorShape((w * h, )), tf.TensorShape(())),
    )


def reader(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    x, y = names[kind]
    with lzma.open(p / (x + '.xz'), mode='rb') as xf:
        assert read32(xf) == 2051
        _, r, c = read32(xf), read32(xf), read32(xf)
        with lzma.open(p / (y + '.xz'), mode='rb') as yf:
            assert read32(yf) == 2049
            _ = read32(yf)
            while True:
                x, y = read32(xf, r * c * 4), read32(yf, 1)
                if x is None or y is None:
                    break
                yield x, int(y)


def read32(f, count=4):
    b = f.read(count)
    if b:
        dt = np.uint8 if count == 1 else np.dtype(np.uint32).newbyteorder('>')
        rs = np.frombuffer(b, dtype=dt)
        if count <= 4:
            return rs[0]
        return np.array(rs, dtype=np.float)


def cached_dset(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    p, r, c = check_images(p / '{}_images'.format(kind))

    def _img(x):
        x = tf.decode_raw(x, tf.uint8)
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [r * c])
        return x / 255.0

    x = tf.FixedLengthRecordDataset(p, r * c, header_bytes=16).map(_img)
    p = check_labels(p / '{}_labels'.format(kind))

    def _lbl(y):
        y = tf.decode_raw(y, tf.uint8)
        y = tf.reshape(y, [])
        return tf.cast(y, tf.int32)

    y = tf.FixedLengthRecordDataset(p, 1, header_bytes=8).map(_lbl)
    return tf.Dataset.zip((x, y))


def np_dset(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    with np.load(p / 'combined') as d:
        x, y = d['x_{}'.format(kind)], d['y_{}'.format(kind)]
    x = x.astype(np.float32) / 255
    x = np.expand_dims(x, -1)
    y = tf.one_hot(y, ps.num_classes)
    return tf.Dataset.from_tensor_slices((x, y))


def np_data(ps, kind):
    p = pth.Path(ps.dir_data) / ps.dset
    with np.load(p / 'combined') as d:
        x, y = d['x_{}'.format(kind)], d['y_{}'.format(kind)]
    r, c = x.shape[1:]
    x = x.reshape((-1, r * c)).astype('float32') / 255
    y = tf.to_categorical(y, ps.num_classes)
    return x, y


def check_images(path):
    with open(path, 'rb') as f:
        if read32(f) != 2051:
            raise ValueError('Invalid magic number {}'.format(f.name))
        read32(f)
        r = read32(f)
        c = read32(f)
    return str(path), r, c


def check_labels(path):
    with open(path, 'rb') as f:
        if read32(f) != 2049:
            raise ValueError('Invalid magic number {}'.format(f.name))
    return str(path)


names = {
    'train': ('train_images', 'train_labels'),
    'test': ('test_images', 'test_labels'),
    'try': ('test_images', 'test_labels'),
}
