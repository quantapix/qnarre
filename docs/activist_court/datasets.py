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

import samples as qs
import utils as qu

td = tf.data
tt = tf.train

vocab = (' ', )
metas = vocab + ('xys', 'ops', 'res')
separs = (';', '[', ']')
vocab += separs
vocab += ('x', 'y', '=', ',', '+', '-', '*')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
masks = ('?', '_')
vocab += masks

tokens = {c: i for i, c in enumerate(vocab)}
tokens.update((c, i) for i, c in enumerate(metas))

SPC = tokens[vocab[0]]
assert SPC == 0
EOS = tokens[separs[-1]]
MSK = tokens[masks[0]]

features = ('enc', 'dec', 'tgt')


@tf.function
def splitter(x):
    # fs = tf.strings.split(x, ':')
    return {f: x[i] for i, f in enumerate(features)}


@tf.function
def tokenizer(d):
    def tokenize(x):
        if chr(x[0]) == '#':
            y = ''.join([chr(c)] for c in x[1:])
            y = [int(v) for v in y.split()]
        else:
            y = [tokens[chr(c)] for c in x]
        return tf.constant(y)

    return {
        k: tf.numpy_function(tokenize, [v], Tout=tf.int32)
        for k, v in d.items()
    }


def sharder(ps, samples=False):
    d = pth.Path('/tmp/q/data')
    for s in range(ps.num_shards):
        s = '{:0>4d}'.format(s)

        def sampler():
            for s in qs.sampler(ps):
                yield [s[g] for g in qs.groups]

        ss = np.array(list(sampler(ps))) if samples else None
        for i, g in enumerate(qs.groups):
            f = d / g
            f.mkdir(parents=True, exist_ok=True)
            f = str(f / f'shard_{s}.tfrecords')
            if ss:
                ds = td.Dataset.from_tensor_slices(ss[:, i])
                yield f, ds.map(splitter).map(tokenizer)
            else:
                yield f


def recorder(samples):
    for s in samples:
        fs = tt.Features(feature={
            f: tt.Feature(int64_list=tt.Int64List(value=s[f]))
            for f in features
        })
        yield tt.Example(features=fs).SerializeToString()


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
    features = {m: tf.io.VarLenFeature(tf.int64) for m in metas[1:]}
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
    for m, s in zip(metas[1:], separs):
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

    def mask(x):
        y = x.flat_values
        e = tf.shape(y)[0]
        s = e // 2
        i = tf.random.uniform([s], maxval=e, dtype=tf.int32)[:, None]
        y = tf.tensor_scatter_nd_update(y, i, tf.fill([s], MSK))
        return x.with_flat_values(y)

    return {
        'encode': tf.concat(ys[:2], axis=1),
        'decode': mask(ys[-1]),
        'target': ys[-1],
        'e_meta': tf.concat(ms[:2], axis=1).flat_values,
        'd_meta': ms[-1].flat_values,
    }


@tf.function
def adapter(d):
    x = tuple(t for k in ('encode', 'decode', 'target')
              for t in (d[k].flat_values, d[k].row_splits))
    y = d['target'].to_tensor()
    return (x + (d['e_meta'], d['d_meta']), (y, y))


def dset_for(ps, adapter=adapter, count=None):
    return load(ps, count=count).map(caster).map(formatter).map(adapter)


params = dict(
    dim_batch=100,
    dim_pool=8 * 1024,
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
