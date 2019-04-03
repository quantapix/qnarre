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
import tensorflow as tf
# import tf_addons as tfa

import qnarre.neura.utils as qu

ks = tf.keras
K = ks.backend
kls = ks.layers


class Squad(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.bert = Bert(PS)
        self.trans = self.bert.transformer

    def build(self, input_shape):
        PS = self.params
        assert self.trans.output_shape[2] == PS.hidden_size
        wi = _get_initer(PS.init_stddev)
        kw = dict(shape=(2, PS.hidden_size), trainable=True)
        self.out_w = self.add_weight(name='out_w', initializer=wi, **kw)
        kw.update(shape=(2, ))
        self.out_b = self.add_weight(name='out_b', initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        y = self.trans(inputs, **kw)
        PS = self.params
        y = tf.reshape(y, [PS.batch_size * PS.max_seq_len, PS.hidden_size])
        y = tf.matmul(y, self.out_w, transpose_b=True)
        y = tf.nn.bias_add(y, self.out_b)
        y = tf.reshape(y, [PS.batch_size, PS.max_seq_len, 2])
        y = tf.transpose(y, [2, 0, 1])
        y = tf.unstack(y, axis=0)
        return y[0], y[1]


class Bert(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.transformer = Transformer(PS)
        ki = _get_initer(PS.init_stddev)
        self.dense = kls.Dense(PS.hidden_size, tf.tanh, kernel_initializer=ki)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        y = self.transformer(inputs, **kw)
        y = tf.squeeze(y[:, 0:1, :], axis=1)
        return self.dense(y, **kw)


class Transformer(kls.Layer):
    typ_embed, pos_embed = None, None

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.embedding = Embedding(PS)
        if PS.num_types:
            self.typ_embed = TypeEmbedding(PS)
        if PS.pos_embed:
            p = PosEmbedding(PS) if PS.pos_embed == 'embedding' else None
            p = PosTiming(PS) if PS.pos_embed == 'timing' else p
            self.pos_embed = p
        self.norm = LayerNorm()
        # tfa.layers.normalizations.LayerNormalization()
        self.drop = kls.Dropout(PS.hidden_drop)
        pre, post = PreProcessor(PS), PostProcessor(PS)
        self.e_stack = EncodeStack(PS, pre, post)
        self.d_stack = DecodeStack(PS, pre, post)
        self.dense = kls.Dense(PS.vocab_size, activation=None)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        src, tgt = inputs
        s, st = src
        s = self.embedding(s, **kw)
        if self.typ_embed:
            s = self.typ_embed([s, st], **kw)
        if self.pos_embed:
            s = self.pos_embed(s, **kw)
        s = self.drop(self.norm(s, **kw), **kw)
        y, a = self.e_stack(s, **kw)
        if tgt:
            t, tt = tgt
            t = self.embedding(t, **kw)
            if self.typ_embed:
                t = self.typ_embed([t, tt], **kw)
            if self.pos_embed:
                t = self.pos_embed(t, **kw)
            t = self.drop(self.norm(t, **kw), **kw)
            y = self.d_stack([y, a, t], **kw)
        return self.dense(y, **kw)  # tf.squeeze(y, axis=2), **kw)

    def get_config(self):
        c = super().get_config()
        c['PS'] = self.PS
        return c


class LayerNorm(kls.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        c = super().get_config()
        c['axis'] = self.axis
        return c

    def build(self, input_shape):
        d = input_shape[-1]
        self.gain = self.add_weight(
            name='gain', shape=(d, ), initializer='ones', trainable=True)
        self.bias = self.add_weight(
            name='bias', shape=(d, ), initializer='zeros', trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        m = K.mean(inputs, axis=self.axis, keepdims=True)
        v = K.mean(K.square(inputs - m), axis=self.axis, keepdims=True)
        e = K.constant(1e-5, dtype=K.floatx())
        y = (inputs - m) / K.sqrt(v + e)
        return self.gain * y + self.bias


class Embedding(kls.Embedding):
    def __init__(self, PS, **_):
        ei = _get_initer(PS.init_stddev)
        er = ks.regularizers.l2(PS.l2_penalty) if PS.l2_penalty else None
        super().__init__(
            input_dim=PS.vocab_size + 1,
            output_dim=PS.hidden_size,
            embeddings_initializer=ei,
            embeddings_regularizer=er,
            mask_zero=True,
            input_length=PS.max_seq_len,
        )


class TypeEmbedding(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        s, t = input_shape
        _, slen, hsize = s
        _, tlen = t
        assert slen == tlen
        PS = self.PS
        ei = _get_initer(PS.init_stddev)
        sh = (PS.num_types, hsize)
        self.embeds = self.add_weight(initializer=ei, shape=sh, trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **_):
        s, t = inputs
        t = K.one_hot(t, self.PS.num_types)
        return s + K.dot(t, self.embeds)


class PosEmbedding(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pos_len = max(PS.max_pos_len, PS.max_seq_len)

    def build(self, input_shape):
        _, slen, hsize = input_shape
        PS = self.PS
        assert slen <= self.pos_len
        ei = _get_initer(PS.init_stddev)
        sh = (self.pos_len, hsize)
        full = self.add_weight(initializer=ei, shape=sh, trainable=True)
        self.embeds = tf.slice(full, [0, 0], [slen, -1])
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + K.expand_dims(self.embeds, 0)


class PosTiming(kls.Layer):
    def __init__(self, _, min_scale=1.0, max_scale=1.0e4, start=0, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.start = start

    def build(self, input_shape):
        _, slen, hsize = input_shape
        assert hsize % 2 == 0
        n = hsize // 2
        s = np.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * K.exp(K.arange(n, dtype=K.floatx()) * -s)
        p = K.arange(slen, dtype=K.floatx()) + self.start
        p = K.expand_dims(p, 1) * K.expand_dims(s, 0)
        p = K.concatenate([K.sin(p), K.cos(p)], axis=1)
        self.timing = K.expand_dims(p, axis=0)
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + self.timing


class Stack(kls.Layer):
    prox_bias = None

    @staticmethod
    def prox(length):
        p = K.arange(length, dtype=K.floatx())
        d = K.expand_dims(p, 0) - K.expand_dims(p, 1)
        return K.expand_dims(K.expand_dims(-tf.log1p(K.abs(d)), 0), 0)

    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post
        self.drop = pre.dropout

    @staticmethod
    def attn_mask(mask):
        m = K.cast(mask, K.floatx())
        m *= qu.min_for(m)
        return K.expand_dims(K.expand_dims(m, axis=1), axis=1)


class EncodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.encode_layers or PS.stacks_layers
        self.encoders = [Encoder(*a, name=f'encoder_{i}') for i in range(n)]

    def build(self, input_shape):
        _, slen, hsize = input_shape
        PS = self.PS
        assert hsize == PS.hidden_size
        if PS.prox_bias:
            self.prox_bias = self.prox(input_shape[1])
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.encoders[-1].output_shape

    def call(self, inputs, mask, **kw):
        s = inputs
        sam = am = self.attn_mask(mask)
        if self.prox_bias:
            sam += self.prox_bias
        if self.PS.pad_remover:
            p = K.cast(K.less(sam, -1.0), K.floatx())
            p = K.squeeze(K.squeeze(p, axis=1), axis=1)
            kw.update(pad_remover=qu.PadRemover(p))
        s = self.drop(s, **kw)
        for e in self.encoders:
            s = e([s, sam], **kw)
        return self.pre(s), am


class DecodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.decode_layers or PS.stacks_layers
        self.decoders = [Decoder(*a, name=f'decoder_{i}') for i in range(n)]

    def build(self, input_shape):
        _, slen, hsize = input_shape
        PS = self.PS
        assert hsize == PS.hidden_size
        if PS.prox_bias:
            self.prox_bias = self.proximity(input_shape[2][1])
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.decoders[-1].output_shape

    def call(self, inputs, mask, **kw):
        s, am, t = inputs
        sam = self.attn_mask(mask[2])
        PS = self.PS
        if PS.causal_self_attn:
            if PS.prepend_mode == 'prepend_inputs_full_attention':
                p = K.cumsum(K.cumsum(sam, axis=1), axis=1)
                p = K.greater(K.expand_dims(p, 1), K.expand_dims(p, 2))
                sam = K.expand_dims(K.cast(p, K.floatx()) * -1e9, 1)
            else:
                ln = K.int_shape(t)[1]
                sh = (1, 1, ln, ln)
                b = qu.ones_band_part(ln, ln, -1, 0, out_shape=sh)
                sam = -1e9 * (1.0 - b)
        else:
            sam = K.expand_dims(K.expand_dims(p, axis=1), axis=1)
        if self.prox_bias:
            sam += self.prox_bias
        t = tf.pad(t, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # t = tf.concat([pad_value, t], axis=1)[:, :-1, :]
        t = self.drop(t, **kw)
        for d in self.decoders:
            t = d([s, am, t, sam], **kw)
        return K.expand_dims(self.pre(t), axis=2)


class Encoder(kls.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.reflection = _attentions[PS.self_attn_type](*a)
        self.fforward = _fforwards[PS.ffn_layer](*a)

    def compute_output_shape(self, _):
        return self.fforward.output_shape

    def call(self, inputs, **kw):
        s, sam = inputs
        s = self.reflection([s, s, sam], **kw)
        return self.fforward(s, **kw)


class Decoder(kls.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.reflection = _attentions[PS.self_attn_type](*a)
        self.attention = _attentions[PS.attn_type](*a)
        self.fforward = _fforwards[PS.ffn_layer](*a, conv_padding='LEFT')

    def compute_output_shape(self, _):
        return self.fforward.output_shape

    def call(self, inputs, **kw):
        s, am, t, sam = inputs
        t = self.reflection([t, t, sam], **kw)
        t = self.attention([t, s, am], **kw)
        return self.fforward(t, **kw)


class Attention(kls.Layer):
    @staticmethod
    def dense_comp(units, **kw):
        return kls.Dense(units, use_bias=False, **kw)

    def __init__(self, PS, pre, post, comp=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        n = PS.attn_heads
        q_size = PS.hidden_size
        assert q_size % n == 0
        comp = comp or self.dense_comp
        ki = _get_initer(PS.init_stddev)
        self.q_comp = comp(q_size, name='Q', kernel_initializer=ki)
        k_size = PS.attn_k_size or PS.hidden_size
        assert k_size % n == 0
        self.k_size = k_size
        self.k_comp = comp(k_size, name='K', kernel_initializer=ki)
        v_size = PS.attn_v_size or PS.hidden_size
        assert v_size % n == 0
        self.v_comp = comp(v_size, name='V', kernel_initializer=ki)
        self.dense = kls.Dense(q_size, use_bias=False, kernel_initializer=ki)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        s, t, am = inputs
        s = self.pre(s)
        q = self.split_heads(self.q_comp(s))
        k = self.split_heads(self.k_comp(t))
        v = self.split_heads(self.v_comp(t))
        y = self.calc_scores(q, k, v, am, **kw)
        y = self.join_heads(y)
        return self.post(s, self.dense(y))

    def split_heads(self, x):
        sh = K.int_shape(x)
        s = sh[-1]
        n = self.PS.attn_heads
        assert s % n == 0
        y = K.reshape(x, sh[:-1] + [n, s // n])
        return K.permute_dimensions(y, [0, 2, 1, 3])

    @staticmethod
    def join_heads(x):
        y = K.permute_dimensions(x, [0, 2, 1, 3])
        sh = K.int_shape(y)
        n, s = sh[-2:]
        return K.reshape(y, sh[:-2] + [n * s])


class ConvComp(kls.Layer):
    dilation_rate = (1, 1)
    padding = 'VALID'

    def __init__(self, filters, ksize, dilation_rate=None, padding=None, **kw):
        super().__init__(**kw)
        assert ksize % 2 == 1
        self.ksize = ksize
        if dilation_rate:
            self.dilation_rate = dilation_rate
        if padding:
            self.padding = padding
        kw = dict(dilation_rate=self.dilation_rate, padding='VALID')
        self.conv = kls.Conv1D(filters, ksize, **kw)

    def call(self, inputs, **kw):
        x = inputs
        if self.padding == 'LEFT':
            sh = K.int_shape(x)
            # h = 2 * (self.ksize // 2) * self.dilation_rate[0]
            # w = 0 if sh[2] == 1 else 2 * (ks[1] // 2) * self.dilation_rate[1]
            # p = tf.constant([[0, 0], [h, 0], [w, 0], [0, 0]])
            # x = tf.pad(x, p)
            # x.set_shape([sh[0], None, None, sh[3]])
        return self.conv(x)


class DotProductAttn(Attention):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        self.drop = kls.Dropout(PS.attn_drop)

    def calc_scores(self, q, k, v, am, **kw):
        y = tf.matmul(q, k, transpose_b=True)
        y *= (self.k_size // self.PS.attn_heads)**-0.5
        y = self.drop(ks.activations.softmax(y + am, **kw), **kw)
        return tf.matmul(y, v)


_attentions = {
    'dot_product': DotProductAttn,
}


class FForward(kls.Layer):
    conv_padding = 'SAME'

    def __init__(self, PS, pre, post, conv_padding=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        if conv_padding:
            self.conv_padding = conv_padding


class DenseDense(FForward):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = _get_act()
        ki = _get_initer(PS.init_stddev)
        self.dense_1 = kls.Dense(
            PS.ffn_units, activation=a, kernel_initializer=ki, use_bias=True)
        self.drop = kls.Dropout(PS.ffn_drop)
        self.dense_2 = kls.Dense(
            PS.hidden_size, kernel_initializer=ki, use_bias=True)

    def call(self, inputs, pad_remover=None, **kw):
        x = inputs
        y = self.pre(x)
        sh = K.int_shape(y)
        if pad_remover:
            y = K.reshape(y, K.concatenate([[-1], sh[2:]], axis=0))
            y = K.expand_dims(pad_remover.remove(y), axis=0)
        y = self.dense_1(y, **kw)
        y = self.dropout(y, **kw)
        y = self.dense_2(y, **kw)
        if pad_remover:
            y = K.reshape(pad_remover.restore(K.squeeze(y, axis=0)), sh)
        return self.post(x, y)


_fforwards = {
    'dense_dense': DenseDense,
}


class Processor(kls.Layer):
    cmd = 'none'

    @staticmethod
    def _dropout(rate, shape, bdims):
        ns, bds = None, [int(i) for i in bdims.split(',') if i]
        if bds:
            n = len(shape)
            bds = [d + n if d < 0 else d for d in bds]
            ns = [1 if i in bds else shape[i] for i in range(n)]
        return kls.Dropout(rate, noise_shape=ns)

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.drop = self._dropout(PS.prepost_drop, (), PS.prepost_bdims)
        self.batch_norm = kls.BatchNormalization(epsilon=PS.norm_epsilon)
        self.norm_type = PS.norm_type
        self.num_groups = PS.num_groups
        self.norm_epsilon = PS.norm_epsilon
        self.group_epsilon = PS.group_epsilon

    def build(self, input_shape):
        filters = input_shape[-1]
        kw = dict(shape=(filters, ), trainable=True)
        self.kern = self.add_weight(name='kern', initializer='ones', **kw)
        self.bias = self.add_weight(name='bias', initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        prev, x = inputs
        if self.cmd != 'none':
            for c in self.cmd:
                if c == 'a':
                    x += prev
                elif c == 'z':
                    self.gamma = tf.get_variable(
                        'gamma', (), initializer=tf.zeros_initializer())
                    x = prev + self.gamma * x
                elif c == 'n':
                    if self.norm_type == 'layer':
                        m = K.mean(x, axis=-1, keepdims=True)
                        v = K.mean(K.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / K.sqrt(v + self.norm_epsilon)
                        x = x * self.kern + self.bias
                    elif self.norm_type == 'group':
                        sh = K.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % self.num_groups == 0
                        gsh = [self.num_groups, sh[-1] // self.num_groups]
                        x = tf.reshape(x, sh[:-1] + gsh)
                        m, v = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
                        x = (x - m) / K.sqrt(v + self.group_epsilon)
                        x = tf.reshape(x, sh) * self.kern + self.bias
                    elif self.norm_type == 'batch':
                        x = self.batch_norm(x, **kw)
                    elif self.norm_type == 'noam':
                        d = K.cast_to_floatx(K.int_shape(x)[-1])
                        x = K.l2_normalize(x, axis=-1) * K.sqrt(d)
                    elif self.norm_type == 'l2':
                        m = K.mean(x, axis=-1, keepdims=True)
                        n = K.sum(K.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / K.sqrt(n + self.epsilon)
                        x = x * self.kern + self.bias
                    else:
                        assert self.norm_type == 'none'
                else:
                    assert c == 'd'
                    x = self.drop(x, **kw)
        return x


class PreProcessor(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.pre_cmd
        assert 'a' not in self.cmd
        assert 'z' not in self.cmd

    def call(self, inputs, **kw):
        super().call([None, inputs], **kw)


class PostProcessor(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.post_cmd


def _get_initer(stddev):
    return ks.initializers.TruncatedNormal(stddev=stddev)


def _gelu_act(x):
    cdf = K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    cdf = (cdf + 1.0) * 0.5
    return x * cdf


def _get_act(name):
    a = name
    if isinstance(a, str):
        a = a.lower()
        if a == "relu":
            a = ks.activations.relu
        elif a == "gelu":
            a = _gelu_act
        elif a == "tanh":
            a = ks.activations.tanh
        else:
            assert a == "linear"
            a = None
    return a
