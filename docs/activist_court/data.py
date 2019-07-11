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


def sampler(ps):
    m, n = ps.max_val, ps.num_samples
    vals = np.random.randint(low=1 - m, high=m, size=(2, n))
    ords = np.random.randint(2, size=(2, n))
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


def sharder(ps, samples=False):
    d = pth.Path('/tmp/q/data')
    d.mkdir(parents=True, exist_ok=True)
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        f = str(d / f'shard_{i}.tfrecords')
        if samples:
            ss = np.array(list(sampler(ps)))
            ds = td.Dataset.from_tensor_slices(ss)
            yield f, ds.map(splitter).map(tokenizer)
        else:
            yield f


def recorder(samples):
    for s in samples:
        features = tt.Features(
            feature={
                'defs': tt.Feature(int64_list=tt.Int64List(value=s['defs'])),
                'op': tt.Feature(int64_list=tt.Int64List(value=s['op'])),
                'res': tt.Feature(int64_list=tt.Int64List(value=s['res'])),
            })
        yield tt.Example(features=features).SerializeToString()


def dump(ps):
    for f, ss in sharder(ps, samples=True):
        print(f'dumping {f}...')
        with tf.io.TFRecordWriter(f) as w:
            for r in recorder(ss):
                w.write(r)
        yield f


def load(ps, files=None, count=None):
    ds = td.TFRecordDataset(files or list(sharder(ps)))
    if count:
        ds = ds.take(count)
    features = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, features))
    return ds.map(lambda x: tf.io.parse_single_example(x, features))


@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}


@tf.function
def formatter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    n = ds.nrows()
    os = tf.RaggedTensor.from_sparse(d['op'])
    tf.debugging.assert_equal(n, os.nrows())
    ss = tf.fill([n, 1], SEP)
    enc = tf.concat([ds, ss, os, ss], axis=1)
    rs = tf.RaggedTensor.from_sparse(d['res'])
    tf.debugging.assert_equal(n, rs.nrows())
    tgt = tf.concat([rs, tf.fill([n, 1], STP)], axis=1)

    def rand_blank(x):
        y = x.flat_values
        mv = tf.shape(y)[0]
        s = mv // 2
        i = tf.random.uniform([s], maxval=mv, dtype=tf.int32)[:, None]
        y = tf.tensor_scatter_nd_update(y, i, tf.zeros([s], dtype=tf.int32))
        return x.with_flat_values(y)

    return {'enc': enc, 'dec': rand_blank(tgt), 'tgt': tgt}


@tf.function
def adapter(d):
    enc, dec, tgt = d['enc'], d['dec'], d['tgt']
    return (
        (
            enc.flat_values,
            enc.row_splits,
            dec.flat_values,
            dec.row_splits,
            tgt.flat_values,
            tgt.row_splits,
        ),
        tgt.to_tensor(),
    )


def dset_for(ps, adapter=adapter):
    return load(ps).map(caster).map(formatter).map(adapter)


params = dict(
    dim_batch=100,
    max_val=100,  # 10000
    num_samples=1000,  # 100000
    num_shards=10,
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main(ps):
    fs = [f for f in dump(ps)]
    ds = load(ps, files=fs).map(caster).map(formatter).map(adapter)
    for i, _ in enumerate(ds):
        pass
    print(f'dumped {i} batches of {ps.dim_batch} samples each')


if __name__ == '__main__':
    ps = Params(**params)
    main(ps)
