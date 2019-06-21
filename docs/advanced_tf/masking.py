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
# !pip install tensorflow==2.0.0-beta0

import numpy as np
import pathlib as pth
import tensorflow as tf

td = tf.data
ks = tf.keras
kl = ks.layers

vocab = ('x', 'y')
vocab += ('+', '-', '*', '=', ',', ':')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

tokens = {k: v for v, k in enumerate(vocab, start=5)}
SEP = tokens[':']


def paths(ps):
    d = pth.Path('/tmp/qnarre/dataset')
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        yield str(d / f'shard_{i}.tfrecords')


@tf.function
def adapter(d):
    ds = tf.cast(d['defs'], tf.int32)
    ds = tf.RaggedTensor.from_sparse(ds)
    ss = tf.fill([ds.nrows(), 1], SEP)
    os = tf.cast(d['op'], tf.int32)
    os = tf.RaggedTensor.from_sparse(os)
    x = tf.concat([ds, ss, os], axis=1)
    y = tf.cast(d['res'], tf.int32)
    y = tf.RaggedTensor.from_sparse(y)
    return x.to_tensor(), y.to_tensor()


def dset_for(ps):
    ds = td.TFRecordDataset(list(paths(ps))).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs))
    return ds.map(adapter)


def model_for(ps):
    m = ks.Sequential()
    m.add(kl.Dense(ps.dim_hidden, input_dim=ps.dim_input, name='in'))
    for i in range(ps.num_layers):
        m.add(Layer(i, ps, name=f'lay_{i}'))
    m.add(kl.Dense(ps.dim_input, name='out'))
    m.compile(optimizer=ps.optimizer(), loss=ps.loss(), metrics=[ps.metrics()])
    print(m.summary())
    return m


params = dict(
    dim_hidden=1000,
    dim_input=100,
    dim_batch=100,
    loss=ks.losses.MeanAbsoluteError,
    metrics=ks.metrics.MeanAbsoluteError,
    num_layers=10,
    num_shards=10,
    optimizer=ks.optimizers.SGD,
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main(_):
    ps = Params(**params)
    for s in dset_for(ps):
        print(s)


if __name__ == '__main__':
    from absl import app
    app.run(main)
