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
# !pip install -U tf-nightly-2.0-preview

import numpy as np
import pathlib as pth
import tensorflow as tf

td = tf.data
tt = tf.train

vocab = (' ', ':', '|')
vocab += ('x', 'y', '=', ',', '+', '-', '*')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

tokens = {c: i for i, c in enumerate(vocab)}

SPC, SEP, STP = [tokens[c] for c in vocab[:3]]
assert SPC == 0

params = dict(
    max_val=10,
    num_samples=4,
    num_shards=3,
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def py_gen(ps):
    m, n = ps.max_val, ps.num_samples
    # x, y vals in defs
    vals = np.random.randint(low=1 - m, high=m, size=(2, n))
    # (x, y) order if 1 in defs [0] and op [1], respectively
    ords = np.random.randint(2, size=(2, n))
    # index of ['+', '-', '*']
    ops = np.array(['+', '-', '*'])
    ops.reshape((1, 3))
    ops = ops[np.random.randint(3, size=n)]
    for i in range(n):
        x, y = vals[:, i]
        res = f'x={x},y={y}:' if ords[0, i] else f'y={y},x={x}:'
        o = ops[i]
        res += (f'x{o}y:' if ords[1, i] else f'y{o}x:')
        if o == '+':
            res += f'{x + y}'
        elif o == '*':
            res += f'{x * y}'
        else:
            assert o == '-'
            res += (f'{x - y}' if ords[1, i] else f'{y - x}')
        yield res


def gen_src(ps):
    ds = td.Dataset.from_generator(
        lambda: py_gen(ps),
        tf.string,
        tf.TensorShape([]),
    )
    return ds


def src_dset(ps):
    ds = np.array(list(py_gen(ps)))
    ds = td.Dataset.from_tensor_slices(ds)
    return ds


@tf.function
def filterer(x):
    r = tf.strings.length(x) < 15
    tf.print(tf.strings.format('filtering {}... ', x) + ('in' if r else 'out'))
    return r


@tf.function
def splitter(x):
    fs = tf.strings.split(x, ':')
    return {'defs': fs[0], 'op': fs[1], 'res': fs[2]}


@tf.function
def tokenizer(d):
    return {
        k: tf.numpy_function(
            lambda x: tf.constant([tokens[chr(c)] for c in x]),
            [v],
            Tout=tf.int32,
        )
        for k, v in d.items()
    }


def shards(ps):
    for _ in range(ps.num_shards):
        yield src_dset(ps).map(splitter).map(tokenizer)


def records(dset):
    for s in dset:
        fs = tt.Features(
            feature={
                'defs': tt.Feature(int64_list=tt.Int64List(value=s['defs'])),
                'op': tt.Feature(int64_list=tt.Int64List(value=s['op'])),
                'res': tt.Feature(int64_list=tt.Int64List(value=s['res'])),
            })
        yield tt.Example(features=fs).SerializeToString()


def dump(ps):
    d = pth.Path('/tmp/q/dataset')
    d.mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(shards(ps)):
        i = '{:0>4d}'.format(i)
        f = str(d / f'shard_{i}.tfrecords')
        print(f'dumping {f}...')
        with tf.io.TFRecordWriter(f) as w:
            for r in records(ds):
                w.write(r)
        yield f


features = {
    'defs': tf.io.VarLenFeature(tf.int64),
    'op': tf.io.VarLenFeature(tf.int64),
    'res': tf.io.VarLenFeature(tf.int64),
}


def files(ps):
    d = pth.Path('/tmp/q/dataset')
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        yield str(d / f'shard_{i}.tfrecords')


def load(ps, files):
    ds = td.TFRecordDataset(files)
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, features))
    return ds.map(lambda x: tf.io.parse_single_example(x, features))


@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}


@tf.function
def adapter(d):
    return [
        tf.sparse.to_dense(d['defs']),
        tf.sparse.to_dense(d['op']),
        tf.sparse.to_dense(d['res']),
    ]


def main(ps):
    for s in py_gen(ps):
        print(s)
    print('Ops on datasets')
    dg = gen_src(ps)
    for s in dg.take(2):
        print(s)
    ds = src_dset(ps)
    for i, s in ds.take(2).concatenate(dg).enumerate():
        print(i, s)
    print('Filter elements')
    for i, s in enumerate(ds.filter(filterer)):
        print(i, s)
    print('Split elements')
    for s in ds.map(splitter).take(1):
        print(s)
    for s in ds.map(splitter).map(tokenizer).take(1):
        print(s)
    fs = [f for f in dump(ps)]
    ps.dim_batch = None
    for i, s in enumerate(load(ps, fs).map(adapter)):
        print(i, s)
    ps.dim_batch = 2
    for i, s in enumerate(load(ps, fs).map(adapter)):
        print(i, s)
    ps.max_val = 100
    ps.num_samples = 1000
    ps.num_shards = 10
    fs = [f for f in dump(ps)]
    ps.dim_batch = 100
    for i, _ in enumerate(load(ps, fs).map(adapter)):
        pass
    print(i)


if __name__ == '__main__':
    ps = Params(**params)
    main(ps)
