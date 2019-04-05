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

    def build(self, input_shape):
        PS = self.PS
        sh = (2, PS.hidden_size)
        wi = _get_initer(PS.init_stddev)
        kw = dict(dtype='float32', trainable=True)
        self.gain = self.add_weight(shape=sh, initializer=wi, **kw)
        self.bias = self.add_weight(shape=2, initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        seq, typ = inputs[:2]
        y = self.bert.transformer([[seq, typ], None], **kw)
        PS = self.PS
        y = K.reshape(y, [PS.batch_size * PS.max_seq_len, PS.hidden_size])
        y = tf.matmul(y, self.gain, transpose_b=True)
        y = K.bias_add(y, self.bias)
        y = K.reshape(y, [PS.batch_size, PS.max_seq_len, 2])
        y = tf.transpose(y, [2, 0, 1])
        y = tf.unstack(y, axis=0)
        return y[0], y[1]


class Bert(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.transformer = Transformer(PS)
        kw = dict(kernel_initializer=_get_initer(PS.init_stddev))
        self.cls_dense = kls.Dense(PS.hidden_size, tf.tanh, **kw)
        self.mlm_dense = kls.Dense(PS.hidden_size, PS.hidden_act, **kw)
        self.norm = LayerNorm()

    def build(self, input_shape):
        PS = self.PS
        sh = PS.vocab_size
        self.bias = self.add_weight(
            shape=sh, initializer='zeros', trainable=True)
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        seq, typ, fit, idx, val, ws = inputs
        y = self.transformer([[seq, typ], None], **kw)
        cls = tf.squeeze(y[:, 0:1, :], axis=1)
        y = self.norm(self.mlm_dense(y, **kw), **kw)
        es = self.transformer.tok_embed.embeddings
        y = tf.matmul(y, es, transpose_b=True)
        y = K.bias_add(y, self.bias)
        y = tf.nn.log_softmax(y, axis=-1)
        return self.cls_dense(cls, **kw)


class Transformer(kls.Layer):
    typ_embed, pos_embed = None, None

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.tok_embed = TokEmbed(PS)
        if PS.token_types:
            self.typ_embed = TypEmbed(PS)
        if PS.pos_embed:
            p = PosEmbed(PS) if PS.pos_embed == 'embed' else None
            p = PosTiming(PS) if PS.pos_embed == 'timing' else p
            self.pos_embed = p
        self.norm = LayerNorm()
        self.drop = kls.Dropout(PS.hidden_drop)
        pre, post = PreProc(PS), PostProc(PS)
        self.e_stack = EncodeStack(PS, pre, post)
        self.d_stack = DecodeStack(PS, pre, post)
        self.dense = kls.Dense(PS.vocab_size, activation=None)

    def compute_output_shape(self, input_shape):
        src, tgt = input_shape
        s, _ = src
        return (s, self.dense.output_shape) if tgt else s

    def call(self, inputs, **kw):
        src, tgt = inputs
        s, typ = src
        s = self.tok_embed(s, **kw)
        if self.typ_embed:
            s = self.typ_embed([s, typ], **kw)
        if self.pos_embed:
            s = self.pos_embed(s, **kw)
        s = self.drop(self.norm(s, **kw), **kw)
        y, attn = self.e_stack(s, **kw)
        if tgt:
            t, typ = tgt
            t = self.tok_embed(t, **kw)
            if self.typ_embed:
                t = self.typ_embed([t, typ], **kw)
            if self.pos_embed:
                t = self.pos_embed(t, **kw)
            t = self.drop(self.norm(t, **kw), **kw)
            y = self.d_stack([y, attn, t], **kw)
            return y, self.dense(y, **kw)
        return y


"""
    def get_config(self):
        c = super().get_config()
        c['PS'] = self.PS
        return c
"""


class TokEmbed(kls.Embedding):
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


class TypEmbed(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        x, typ = input_shape
        _, xlen, hsize = x
        _, tlen = typ
        assert xlen == tlen
        PS = self.PS
        sh = (PS.token_types, hsize)
        wi = _get_initer(PS.init_stddev)
        self.gain = self.add_weight(shape=sh, initializer=wi, trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **_):
        x, typ = inputs
        typ = K.one_hot(typ, self.PS.token_types)
        return x + K.dot(typ, self.gain)


class PosEmbed(kls.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        _, xlen, hsize = input_shape
        PS = self.PS
        plen = max(PS.max_pos_len, PS.max_seq_len)
        assert xlen <= plen
        sh = (plen, hsize)
        wi = _get_initer(PS.init_stddev)
        b = self.add_weight(shape=sh, initializer=wi, trainable=True)
        self.bias = tf.slice(b, [0, 0], [xlen, -1])
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + K.expand_dims(self.bias, 0)


class PosTiming(kls.Layer):
    start = 0
    min_scale = 1.0
    max_scale = 1.0e4

    def __init__(self, _, start=None, min_scale=None, max_scale=None, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        if start:
            self.start = start
        if min_scale:
            self.min_scale = float(min_scale)
        if max_scale:
            self.max_scale = float(max_scale)

    def build(self, input_shape):
        _, xlen, hsize = input_shape
        assert hsize % 2 == 0
        n = hsize // 2
        s = np.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * K.exp(K.arange(n, dtype=K.floatx()) * -s)
        p = K.arange(xlen, dtype=K.floatx()) + self.start
        p = K.expand_dims(p, 1) * K.expand_dims(s, 0)
        p = K.concatenate([K.sin(p), K.cos(p)], axis=1)
        self.bias = K.expand_dims(p, axis=0)
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + self.bias


class LayerNorm(kls.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True

    def build(self, input_shape):
        kw = dict(shape=input_shape[-1], trainable=True)
        self.gain = self.add_weight(initializer='ones', **kw)
        self.bias = self.add_weight(initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **_):
        x = inputs
        m = K.mean(x, axis=-1, keepdims=True)
        v = K.mean(K.square(x - m), axis=-1, keepdims=True)
        e = K.constant(1e-5, dtype=K.floatx())
        y = (x - m) / K.sqrt(v + e)
        return self.gain * y + self.bias


class Stack(kls.Layer):
    prox_bias = None

    @staticmethod
    def proximity(slen):
        p = K.arange(slen, dtype=K.floatx())
        p = K.expand_dims(p, 0) - K.expand_dims(p, 1)
        return K.expand_dims(K.expand_dims(-tf.log1p(K.abs(p)), 0), 0)

    @staticmethod
    def attn_mask(mask):
        f = K.floatx()
        fmin = tf.float16.min if f == 'float16' else tf.float32.min
        m = K.cast(mask, f) * fmin
        return K.expand_dims(K.expand_dims(m, axis=1), axis=1)

    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post


class EncodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.encode_layers or PS.stack_layers
        self.encs = [Encoder(*a, name=f'enc_{i}') for i in range(n)]

    def build(self, input_shape):
        if self.PS.prox_bias:
            self.prox_bias = self.proximity(input_shape[1])
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.encoders[-1].output_shape

    def call(self, inputs, mask, **kw):
        s = inputs
        sam = am = self.attn_mask(mask)
        if self.prox_bias:
            sam += self.prox_bias
        if self.PS.pad_remover:
            kw.update(pad_remover=qu.PadRemover(mask))
        y = self.pre.drop(s, **kw)
        for e in self.encs:
            y = e([y, sam], **kw)
        return self.post([s, y], **kw), am


class DecodeStack(Stack):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        a = (PS, self.pre, self.post)
        n = PS.decode_layers or PS.stack_layers
        self.decs = [Decoder(*a, name=f'dec_{i}') for i in range(n)]

    def build(self, input_shape):
        if self.PS.prox_bias:
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
        if self.prox_bias:
            sam += self.prox_bias
        t = tf.pad(t, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # t = tf.concat([pad_value, t], axis=1)[:, :-1, :]
        y = self.pre.drop(t, **kw)
        for d in self.decs:
            y = d([s, am, y, sam], **kw)
        return K.expand_dims(self.post([t, y], **kw), axis=2)


class Encoder(kls.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = _attns[PS.self_attn_type](*a)
        self.ffn = _ffns[PS.ffn_layer](*a)

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        s, sam = inputs
        y = self.refl([s, s, sam], **kw)
        return self.ffn(y, **kw)


class Decoder(kls.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.refl = _attns[PS.self_attn_type](*a)
        self.attn = _attns[PS.attn_type](*a)
        self.ffn = _ffns[PS.ffn_layer](*a, conv_pad='LEFT')

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        s, am, t, sam = inputs
        y = self.refl([t, t, sam], **kw)
        y = self.attn([y, s, am], **kw)
        return self.fforward(y, **kw)


class Attention(kls.Layer):
    def __init__(self, PS, pre, post, comp=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        self.comp = comp or self.dense_comp
        self.dense = self.dense_comp(PS.hidden_size)

    def build(self, input_shape):
        src, tgt, _ = input_shape
        _, slen, hsize = src
        _, tlen, hs2 = tgt
        assert hsize == hs2
        PS = self.PS
        n = PS.attn_heads
        assert hsize % n == 0
        self.q_comp = self.comp(hsize, name='Q')
        k_size = PS.attn_k_size or hsize
        assert k_size % n == 0
        self.k_size = k_size
        self.k_comp = self.comp(k_size, name='K')
        v_size = PS.attn_v_size or hsize
        assert v_size % n == 0
        self.v_comp = self.comp(v_size, name='V')
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        s, t, am = inputs
        s = self.pre(s, **kw)
        q = self.split_heads(self.q_comp(s, **kw))
        k = self.split_heads(self.k_comp(t, **kw))
        v = self.split_heads(self.v_comp(t, **kw))
        y = self.calc_scores(q, k, v, am, **kw)
        y = self.join_heads(y)
        return self.post([s, self.dense(y)], **kw)

    def dense_comp(self, units, **kw):
        ki = _get_initer(self.PS.init_stddev)
        return kls.Dense(units, use_bias=False, kernel_initializer=ki, **kw)

    def split_heads(self, x):
        sh = K.int_shape(x)
        n = self.PS.attn_heads
        y = K.reshape(x, (-1, sh[1], n, sh[-1] // n))
        return K.permute_dimensions(y, [0, 2, 1, 3])

    @staticmethod
    def join_heads(x):
        y = K.permute_dimensions(x, [0, 2, 1, 3])
        sh = K.int_shape(y)
        return K.reshape(y, (-1, sh[1], sh[2] * sh[3]))


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


class DotAttn(Attention):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        self.drop = kls.Dropout(PS.attn_drop)

    def calc_scores(self, q, k, v, am, **kw):
        y = tf.matmul(q, k, transpose_b=True)
        y *= (self.k_size // self.PS.attn_heads)**-0.5
        y = self.drop(ks.activations.softmax(y + am, **kw), **kw)
        return tf.matmul(y, v)


_attns = {
    'dot_attn': DotAttn,
}


class FForward(kls.Layer):
    conv_pad = 'SAME'

    def __init__(self, PS, pre, post, conv_pad=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        if conv_pad:
            self.conv_pad = conv_pad


class DenseDense(FForward):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        ac = _get_act(PS.ffn_act)
        ki = _get_initer(PS.init_stddev)
        kw = dict(kernel_initializer=ki, use_bias=True)
        self.dense1 = kls.Dense(PS.ffn_units, activation=ac, **kw)
        self.drop = kls.Dropout(PS.ffn_drop)
        self.dense2 = kls.Dense(PS.hidden_size, **kw)

    def call(self, inputs, pad_remover=None, **kw):
        x = inputs
        y = self.pre(x, **kw)
        sh = K.int_shape(y)
        if pad_remover:
            y = K.reshape(y, K.concatenate([[-1], sh[2:]], axis=0))
            y = K.expand_dims(pad_remover.remove(y), axis=0)
        y = self.dense1(y, **kw)
        y = self.drop(y, **kw)
        y = self.dense2(y, **kw)
        if pad_remover:
            y = K.reshape(pad_remover.restore(K.squeeze(y, axis=0)), sh)
        return self.post([x, y], **kw)


_ffns = {
    'dense_dense': DenseDense,
}


class Processor(kls.Layer):
    cmd = None

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
        self.supports_masking = True
        self.PS = PS
        self.drop = self._dropout(PS.prepost_drop, (), PS.prepost_bdims)
        self.batch = kls.BatchNormalization(epsilon=PS.norm_epsilon)

    def build(self, input_shape):
        _, x = input_shape
        kw = dict(shape=x[-1], trainable=True)
        self.gain = self.add_weight(initializer='ones', **kw)
        self.bias = self.add_weight(initializer='zeros', **kw)
        # kw.update(shape=())
        self.gamma = self.add_weight(initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        prev, x = inputs
        if self.cmd:
            PS = self.PS
            for c in self.cmd:
                if c == 'a':
                    x += prev
                elif c == 'z':
                    x = prev + self.gamma * x
                elif c == 'n':
                    if PS.norm_type == 'layer':
                        m = K.mean(x, axis=-1, keepdims=True)
                        v = K.mean(K.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / K.sqrt(v + PS.norm_epsilon)
                        x = x * self.gain + self.bias
                    elif PS.norm_type == 'batch':
                        x = self.batch(x, **kw)
                    elif PS.norm_type == 'l2':
                        m = K.mean(x, axis=-1, keepdims=True)
                        n = K.sum(K.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / K.sqrt(n + PS.norm_epsilon)
                        x = x * self.gain + self.bias
                    elif PS.norm_type == 'group':
                        sh = K.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % PS.num_groups == 0
                        gsh = (PS.num_groups, sh[-1] // PS.num_groups)
                        x = K.reshape(x, sh[:-1] + gsh)
                        m, v = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
                        x = (x - m) / K.sqrt(v + PS.group_epsilon)
                        x = K.reshape(x, sh) * self.gain + self.bias
                    elif PS.norm_type == 'noam':
                        d = K.cast_to_floatx(K.int_shape(x)[-1])
                        x = K.l2_normalize(x, axis=-1) * K.sqrt(d)
                    else:
                        assert PS.norm_type == 'none'
                else:
                    assert c == 'd'
                    x = self.drop(x, **kw)
        return x


class PreProc(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.pre_cmd
        assert 'a' not in self.cmd
        assert 'z' not in self.cmd

    def build(self, input_shape):
        return super().build((None, input_shape))

    def compute_output_shape(self, input_shape):
        return input_shape,

    def call(self, inputs, **kw):
        return super().call([None, inputs], **kw)


class PostProc(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.post_cmd


def _get_initer(stddev):
    return ks.initializers.TruncatedNormal(stddev=stddev)


def _gelu(x):
    c = K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    c = (c + 1.0) * 0.5
    return x * c


def _get_act(name):
    if isinstance(name, str):
        n = name.lower()
        if n == 'gelu':
            return _gelu
        elif n == 'relu':
            return ks.activations.relu
        elif n == 'tanh':
            return ks.activations.tanh
        else:
            assert n == 'linear'
            name = None
    return name
