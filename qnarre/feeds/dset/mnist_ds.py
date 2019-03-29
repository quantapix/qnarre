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

import numpy as np
import pathlib as pth
import tensorflow as tf

from tensorflow import keras as ks


def dataset(kind, params):
    path = pth.Path(params.data_dir)
    p, r, c = _check_images(path / '{}_images'.format(kind))

    def _img(x):
        x = tf.io.decode_raw(x, tf.uint8)
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [r * c])
        return x / 255.0

    x = tf.data.FixedLengthRecordDataset(p, r * c, header_bytes=16).map(_img)
    p = _check_labels(path / '{}_labels'.format(kind))

    def _lbl(y):
        y = tf.io.decode_raw(y, tf.uint8)
        y = tf.reshape(y, [])
        return tf.cast(y, tf.int32)

    y = tf.data.FixedLengthRecordDataset(p, 1, header_bytes=8).map(_lbl)
    return tf.data.Dataset.zip((x, y))


def np_dataset(kind, params):
    path = pth.Path(params.data_dir)
    with np.load(path / 'combined') as d:
        x, y = d['x_{}'.format(kind)], d['y_{}'.format(kind)]
    x = x.astype(np.float32) / 255
    x = np.expand_dims(x, -1)
    y = tf.one_hot(y, params.num_classes)
    return tf.data.Dataset.from_tensor_slices((x, y))


def np_data(kind, params):
    path = pth.Path(params.data_dir)
    with np.load(path / 'combined') as d:
        x, y = d['x_{}'.format(kind)], d['y_{}'.format(kind)]
    r, c = x.shape[1:]
    x = x.reshape((-1, r * c)).astype('float32') / 255
    y = ks.utils.to_categorical(y, params.num_classes)
    return x, y


def _check_images(path):
    with open(path, 'rb') as f:
        if _read32(f) != 2051:
            raise ValueError('Invalid magic number {}'.format(f.name))
        _read32(f)
        r = _read32(f)
        c = _read32(f)
    return str(path), r, c


def _check_labels(path):
    with open(path, 'rb') as f:
        if _read32(f) != 2049:
            raise ValueError('Invalid magic number {}'.format(f.name))
    return str(path)


def _read32(bstream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bstream.read(4), dtype=dt)[0]
