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
import multiprocessing as mp

import samples as qs

td = tf.data
tt = tf.train

vocab = tuple(' ')
metas = vocab + tuple('XYS OPS RES'.split())
metas += tuple('ABCDEFGHIJ')
separs = tuple(',;[]|')
vocab += separs
vocab += tuple('xy=$+-*')
vocab += tuple('0123456789')
masks = tuple('?_')
vocab += masks

tokens = {c: i for i, c in enumerate(vocab)}
tokens.update((c[0], i) for i, c in enumerate(metas))

SPC = tokens[vocab[0]]
assert SPC == 0
EOS = tokens[separs[-1]]
MSK = tokens[masks[0]]

features = tuple('grp enc dec tgt emt dmt out'.split())

GRP, ENC, DEC, TGT, EMT, DMT, OUT = features


def sampler(ps, groups):
    def to_meta(g, x):
        # print('to_metas x', x)
        g = chr(ord('A') + g)
        m, y = '', x.split(';')
        if len(y) > 1:
            m = 'X' * (len(y[0]) + 1)
            y = y[1:]
        y = y[0].split('[')
        y2 = y[0].split('|')
        m += 'O' * len(y2[0])
        if len(y2) > 1:
            m += g * (len(y2[1]) + 1)
        if len(y) > 1:
            y2 = y[1].split('|')
            m += 'R' * (len(y2[0]) + 1)
            if len(y2) > 1:
                m += g * (len(y2[1]) + 1)
        assert len(x) == len(m)
        # print('to_metas m', m)
        return m

    for s in qs.sampler(ps):

        def to_features(g):
            fs = s[g]
            g = qs.groups.index(g)
            e, d, t, o = fs[ENC], fs[DEC], fs[TGT], fs.get(OUT, '')
            d2 = t if '?' in d else d
            return [f'#{g}', e, d, t, to_meta(g, e), to_meta(g, d2), o]

        yield [to_features(g) for g in groups]


@tf.function
def splitter(x):
    return {f: x[i] for i, f in enumerate(features)}


@tf.function
def tokenizer(d):
    def to_tokens(x):
        if len(x) > 0 and chr(x[0]) == '#':
            y = ''.join([chr(c) for c in x[1:]])
            y = [int(v) for v in y.split()]
        else:
            y = [tokens[chr(c)] for c in x]
        return tf.constant(y, tf.int32)

    return {
        k: tf.numpy_function(to_tokens, [v], Tout=tf.int32)
        for k, v in d.items()
    }


def recorder(examples):
    for e in examples:
        fs = tt.Features(feature={
            f: tt.Feature(int64_list=tt.Int64List(value=e[f]))
            for f in features
        })
        yield tt.Example(features=fs).SerializeToString()


def shards_for(ps, root, groups):
    for s in range(ps.num_shards):
        s = '{:0>4d}'.format(s)
        for i, g in enumerate(groups):
            f = root / g
            f.mkdir(parents=True, exist_ok=True)
            f = str(f / f'shard_{s}.tfrecords')
            yield f, i


def write(shard):
    ps, gs, f, i = shard
    ss = np.array(list(sampler(ps, gs)))
    ds = td.Dataset.from_tensor_slices(ss[:, i])
    ds = ds.map(splitter, -1).map(tokenizer, -1)
    print(f'dumping {f}...')
    with tf.io.TFRecordWriter(f) as w:
        for r in recorder(ds):
            w.write(r)


def dump(ps, root=None, group=None):
    r = pth.Path(root or '/tmp/q/data')
    gs = qs.groups if group is None else [group]
    shards = [(ps, gs, f, i) for f, i in shards_for(ps, r, gs)]
    with mp.Pool() as pool:
        pool.map(write, shards)
    return [s[2] for s in shards]


def load(ps, root=None, group=None, shards=None, count=None):
    if shards is None:
        r = pth.Path(root or '/tmp/q/data')
        gs = qs.groups if group is None else [group]
        shards = [s[0] for s in shards_for(ps, r, gs)]
    ds = td.TFRecordDataset(shards)
    if count:
        ds = ds.take(count)
    fs = {f: tf.io.VarLenFeature(tf.int64) for f in features}
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, fs), -1)
    return ds.map(lambda x: tf.io.parse_single_example(x, fs), -1)


@tf.function
def adapter(d, group=None):
    d = {f: tf.cast(d[f], tf.int32) for f in features}
    d = {f: tf.RaggedTensor.from_sparse(d[f]) for f in features}
    # x = tuple(d[GRP].to_tensor())
    x = tuple(t for f in (ENC, DEC, TGT)
              for t in (d[f].flat_values, tf.cast(d[f].row_splits, tf.int32)))
    x += tuple(d[f].flat_values for f in (EMT, DMT))
    if group in (qs.QAS, qs.FIX):
        y = (d[OUT].to_tensor(), )
    else:
        y = (d[TGT].to_tensor(), )
    return x, y


def dset_for(ps, root=None, group=None, adapter=adapter, count=None):
    ds = load(ps, root, group, count=count)
    ds = ds.map(lambda x: adapter(x, group), -1)
    return ds.shuffle(1000)


if __name__ == '__main__':
    np.random.seed(12345)
    import utils as qu
    ps = dict(
        dim_batch=5,
        dim_pool=10,
        max_val=1000,
        num_samples=20,
        num_shards=3,
    )
    ps = qu.Params(**ps)
    ss = [s for s in dump(ps)]
    ds = load(ps, shards=ss).map(adapter, -1)
    for i, _ in enumerate(ds):
        pass
    print(f'dumped {i + 1} batches of {ps.dim_batch} samples each')
