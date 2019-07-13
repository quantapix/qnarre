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

import utils as qu

td = tf.data
tt = tf.train

metas = ('defs', 'ops', 'res')
separs = (':', ';', '|')

vocab = (' ', )
vocab += separs
vocab += ('x', 'y', '=', ',', '+', '-', '*')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

tokens = {c: i for i, c in enumerate(vocab)}
tokens.update((c, i) for i, c in enumerate(metas, start=len(tokens)))

SPC = tokens[vocab[0]]
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
    return {m: fs[i] for i, m in enumerate(metas)}


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
        features = tt.Features(feature={
            m: tt.Feature(int64_list=tt.Int64List(value=s[m]))
            for m in metas
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
    features = {m: tf.io.VarLenFeature(tf.int64) for m in metas}
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, features))
    return ds.map(lambda x: tf.io.parse_single_example(x, features))


@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}


@tf.function
def formatter(d):
    n, ys, ms = None, [], []
    for m, s in zip(metas, separs):
        y = tf.RaggedTensor.from_sparse(d[m])
        if n is None:
            n = y.nrows()
        else:
            tf.debugging.assert_equal(n, y.nrows())
        y = tf.concat([y, tf.fill([n, 1], tokens[s])], axis=1)
        ys.append(y)
        rs = y.row_lengths()
        y = tf.fill([tf.reduce_sum(rs)], tokens[m])
        y = tf.RaggedTensor.from_row_lengths(y, rs)
        ms.append(y)

    def blank(x):
        y = x.flat_values
        mv = tf.shape(y)[0]
        s = mv // 2
        i = tf.random.uniform([s], maxval=mv, dtype=tf.int32)[:, None]
        y = tf.tensor_scatter_nd_update(y, i, tf.zeros([s], dtype=tf.int32))
        return x.with_flat_values(y)

    return {
        'encode': tf.concat(ys[:2], axis=1),
        'decode': blank(ys[-1]),
        'target': ys[-1],
        'e_meta': tf.concat(ms[:2], axis=1).flat_values,
        'd_meta': ms[-1].flat_values,
    }


@tf.function
def adapter(d):
    return (
        tuple(t for k in ('encode', 'decode', 'target')
              for t in (d[k].flat_values, d[k].row_splits)) +
        (d['e_meta'], d['d_meta']),
        d['target'].to_tensor(),
    )


def dset_for(ps, adapter=adapter):
    return load(ps).map(caster).map(formatter).map(adapter)


params = dict(
    dim_batch=100,
    max_val=100,  # 10000
    num_samples=1000,  # 100000
    num_shards=10,
)


def main(ps):
    fs = [f for f in dump(ps)]
    ds = load(ps, files=fs).map(caster).map(formatter).map(adapter)
    for i, _ in enumerate(ds):
        pass
    print(f'dumped {i} batches of {ps.dim_batch} samples each')


if __name__ == '__main__':
    ps = qu.Params(**params)
    main(ps)
