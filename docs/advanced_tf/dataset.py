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
# ks = tf.keras
# kl = ks.layers

# complex input pipelines from simple, reusable pieces
# a pipeline starts with a "src_dset" and chains "transformations" to it

# example - fascinating https://arxiv.org/pdf/1812.02825.pdf
# num_samples of "x=-12,y=24:y+x:12" w/ "defs", "op" and "res"
# vars: x, y, ops: +, -, *, vals: [-max_val, max_val]

vocab = ('x', 'y')
vocab += ('+', '-', '*', '=', ',', ':')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# pipeline is parametric

params = dict(
    max_val=10,
    num_samples=4,
    num_shards=3,
)


class Params:
    # without error-prone string names
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def py_gen(ps):
    # generate samples
    m, n = ps.max_val, ps.num_samples
    # x, y vals in defs
    vals = np.random.randint(low=-m, high=m + 1, size=(2, n))
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


# tf.data.Dataset - abstraction for a sequence of elements
# each element is one or more Tensors

# can be consumed as iterables or as aggregatables (reduce)


def gen_src(ps):
    # from_generator
    ds = td.Dataset.from_generator(
        lambda: py_gen(ps),
        tf.string,
        tf.TensorShape([]),
    )
    return ds


def src_dset(ps):
    # range, from_tensor_slices, from_tensors
    # also TextLineDataset
    ds = np.array(list(py_gen(ps)))
    ds = td.Dataset.from_tensor_slices(ds)
    return ds


# inside a tf.function or eagerly
@tf.function
def filterer(x):
    r = tf.strings.length(x) < 15
    tf.print(tf.strings.format('filtering {}... ', x) + ('in' if r else 'out'))
    return r


@tf.function
def splitter(x):
    fs = tf.strings.split(x, ':')
    return {'defs': fs[0], 'op': fs[1], 'res': fs[2]}


tokens = {k: v for v, k in enumerate(vocab, start=5)}


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


# potentially large
# transparent performance is key


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
        p = str(d / f'shard_{i}.tfrecords')
        print(f'dumping {p}...')
        with tf.io.TFRecordWriter(p) as w:
            for r in records(ds):
                w.write(r)
        yield p


features = {
    'defs': tf.io.VarLenFeature(tf.int64),
    'op': tf.io.VarLenFeature(tf.int64),
    'res': tf.io.VarLenFeature(tf.int64),
}


def load(ps, paths):
    ds = td.TFRecordDataset(paths)
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, features))
    return ds.map(lambda x: tf.io.parse_single_example(x, features))


@tf.function
def adapter(d):
    return [
        tf.sparse.to_dense(d['defs']),
        tf.sparse.to_dense(d['op']),
        tf.sparse.to_dense(d['res']),
    ]


def main(_):
    ps = Params(**params)
    for s in py_gen(ps):
        print(s)
    # cache, concatenate, enumerate, reduce, repeat, shuffle, skip, take, zip
    print('Ops on datasets')
    dg = gen_src(ps)
    for s in dg.take(2):
        print(s)
    ds = src_dset(ps)
    for i, s in ds.take(2).concatenate(dg).enumerate():
        print(i, s)
    # filter
    print('Filter dataset elements')
    for i, s in enumerate(ds.filter(filterer)):
        print(i, s)
    # map
    for s in ds.map(splitter).take(1):
        print(s)
    for s in ds.map(splitter).map(tokenizer).take(1):
        print(s)
    fs = [f for f in dump(ps)]
    ps.dim_batch = None
    for i, s in enumerate(load(ps, fs).map(adapter)):
        print(i, s)
    # batch, padded_batch
    ps.dim_batch = 2
    for i, s in enumerate(load(ps, fs).map(adapter)):
        print(i, s)
    # apply, flat_map, interleave, prefetch
    ps.max_val = 100
    ps.num_samples = 1000
    ps.num_shards = 10
    fs = [f for f in dump(ps)]
    ps.dim_batch = 100
    for i, _ in enumerate(load(ps, fs).map(adapter)):
        print(i)


if __name__ == '__main__':
    from absl import app
    app.run(main)
