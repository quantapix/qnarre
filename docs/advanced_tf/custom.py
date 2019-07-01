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
# export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

import tensorflow as tf
import dataset as qd
import layers as ql

ks = tf.keras
kl = ks.layers


def dump_dset(ps):
    ps.max_val = 10000
    ps.num_samples = 1000  # 100000
    ps.num_shards = 10
    fs = [f for f in qd.dump(ps)]
    ps.dim_batch = 100
    for i, _ in enumerate(qd.load(ps, fs).map(adapter)):
        pass
    print(f'dumped {i} batches of {ps.dim_batch} samples each')
    return fs


@tf.function
def formatter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    n = ds.nrows()
    os = tf.RaggedTensor.from_sparse(d['op'])
    tf.debugging.assert_equal(n, os.nrows())
    ss = tf.fill([n, 1], qd.SEP)
    enc = tf.concat([ds, ss, os, ss], axis=1)
    rs = tf.RaggedTensor.from_sparse(d['res'])
    tf.debugging.assert_equal(n, rs.nrows())
    tgt = tf.concat([rs, tf.fill([n, 1], qd.STP)], axis=1)

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
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.take(100).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(formatter).map(adapter)


class ToRagged(kl.Layer):
    @tf.function
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys


class Frames(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)  # , dynamic=True)
        s = (ps.dim_batch, ps.width_enc)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', shape=s, **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x
        ye = tf.concat([self.prev, xe], axis=1)
        el = tf.cast(xe.row_lengths(), dtype=tf.int32)
        ye = tf.gather_nd(ye, self.calc_idxs(el))
        c = self.ps.width_dec - xd.bounding_shape(axis=1, out_type=tf.int32)
        yd = tf.pad(xd.to_tensor(), [[0, 0], [0, c]])
        dl = tf.cast(xd.row_lengths(), dtype=tf.int32)
        p = tf.concat([ye, xt], axis=1)
        tl = tf.cast(xt.row_lengths(), dtype=tf.int32)
        p = tf.gather_nd(p, self.calc_idxs(tl))
        self.prev.assign(p)
        return [ye, el, yd, dl]

    def calc_idxs(self, lens):
        b, w = self.ps.dim_batch, self.ps.width_enc
        y = tf.broadcast_to(tf.range(b)[:, None], [b, w])
        i = tf.range(w)[None, ] + lens[:, None]
        y = tf.stack([y, i], axis=2)
        return y


class Embed(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight('emb', shape=s)

    @tf.function
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        y *= y.shape[-1]**0.5
        return [y, lens]


class Encode(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_enc
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y = x
        for e in self.encs:
            y = e(y)
        return y


class Decode(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_dec
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for d in self.decs:
            y = d(y + [ye])
        return y


class Debed(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.dbd = Dense(self, 'dbd', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.dbd(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y


class Encoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y = x
        y = self.reflect(y + [y[0]])
        y = self.conclude(y)
        return y


class Decoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.consider = Attention(layer, 'cnsd')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y, ye = x[:-1], x[-1]
        y = self.reflect(y + [y[0]])
        y = self.consider(y + [ye])
        y = self.conclude(y)
        return y


class Attention(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        h = layer.ps.dim_hidden
        self.scale = 1 / (h**0.5)
        with self.name_scope:
            self.q = layer.add_weight('q', shape=(h, h))
            self.k = layer.add_weight('k', shape=(h, h))
            self.v = layer.add_weight('v', shape=(h, h))

    @tf.function
    def __call__(self, x):
        x, lens, ctx = x
        off = tf.math.reduce_max(lens)
        q = tf.einsum('bni,ij->bnj', x[:, -off:, :], self.q)
        k = tf.einsum('bni,ij->bnj', ctx, self.k)
        y = tf.einsum('bni,bmi->bnm', q, k)
        # use lens
        y = tf.nn.softmax(y * self.scale)
        v = tf.einsum('bni,ij->bnj', ctx, self.v)
        y = tf.einsum('bnm,bmi->bni', y, v)
        y = tf.concat([x[:, :-off, :], y], axis=1)
        return [y, lens]


class Conclusion(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        self.layer = layer
        ps = layer.ps
        w = layer.width * ps.dim_hidden
        with self.name_scope:
            s = [w, ps.dim_dense]
            self.inflate = Dense(layer, 'infl', s, activation='relu')
            s = [ps.dim_dense, w]
            self.deflate = Dense(layer, 'defl', s, bias=False)

    @tf.function
    def __call__(self, x):
        y, lens = x
        w = self.layer.width
        d = self.layer.ps.dim_hidden
        y = tf.reshape(y, [-1, w * d])
        y = self.inflate(y)
        y = self.deflate(y)
        y = tf.reshape(y, [-1, w, d])
        return [y, lens]


class Dense(ql.Dense):
    @tf.function
    def __call__(self, x):
        return super().__call__(x)


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = ToRagged()(x)
    y = Frames(ps)(y)
    embed = Embed(ps)
    ye = Encode(ps)(embed(y[:2]))
    yd = Decode(ps)(embed(y[2:]) + [ye[0]])
    y = Debed(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


params = dict(
    dim_batch=5,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=5,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    width_dec=15,
    width_enc=25,
)

if __name__ == '__main__':
    ps = qd.Params(**params)
    import advanced_tf.masking as qm
    qm.main_graph(ps, dset_for(ps), model_for(ps))
    # import advanced_tf.ragged as qr
    # qr.main_eager(ps, dset_for(ps), model_for(ps))
