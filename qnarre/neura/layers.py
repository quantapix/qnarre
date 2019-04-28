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
# https://arxiv.org/pdf/1701.06548.pdf
# https://arxiv.org/pdf/1607.06450.pdf
# https://arxiv.org/pdf/1606.08415.pdf

import numpy as N
import tensorflow as T
# import tf_addons as tfa

import qnarre.neura.utils as U

KS = T.keras
K = KS.backend
KL = KS.layers


class Squad(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        self.PS = PS
        self.bert = Bert(PS)

    def build(self, input_shape):
        _, slen, hsize = input_shape[0]
        PS = self.PS
        assert slen == PS.max_seq_len
        assert hsize == PS.hidden_size
        kw = dict(initializer=_get_initer(PS.init_stddev))
        self.gain = self.add_weight(shape=(2, hsize), **kw)
        kw.update(initializer='zeros')
        self.bias = self.add_weight(shape=2, **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        y = self.bert.transformer([inputs, None], **kw)
        y = K.bias_add(T.matmul(y, self.gain, transpose_b=True), self.bias)
        return list(T.unstack(T.transpose(y, [2, 0, 1]), axis=0))


class SquadLoss(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.slen = PS.max_seq_len

    def build(self, input_shape):
        PS = self.PS
        wi = _get_initer(PS.init_stddev)
        kw = dict(initializer=wi, dtype='float32', trainable=True)
        self.gain = self.add_weight(shape=(2, PS.hidden_size), **kw)
        kw.update(initializer='zeros')
        self.bias = self.add_weight(shape=2, **kw)
        return super().build(input_shape)

    def call(self, inputs, **_):
        span, pred = inputs

        def _loss(i):
            y = T.nn.log_softmax(pred[i], axis=-1)
            y = K.one_hot(span[:, i], self.slen) * y
            return -K.mean(K.sum(y, axis=-1))

        self.add_loss((_loss(0) + _loss(1)) / 2.0)
        return pred


class Bert(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.transformer = Transformer(PS)
        kw = dict(kernel_initializer=_get_initer(PS.init_stddev))
        self.pool = KL.Dense(PS.hidden_size, T.tanh, **kw)
        self.mlm_dense = KL.Dense(PS.hidden_size, PS.hidden_act, **kw)
        self.embed = self.transformer.tok_embed.embeddings
        self.norm = LayerNorm()

    def build(self, input_shape):
        PS = self.PS
        wi = _get_initer(PS.init_stddev)
        kw = dict(initializer=wi, dtype='float32', trainable=True)
        self.gain = self.add_weight(shape=(2, PS.hidden_size), **kw)
        kw.update(initializer='zeros')
        self.mlm_bias = self.add_weight(shape=PS.vocab_size, **kw)
        self.bias = self.add_weight(shape=2, **kw)
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.dense.output_shape

    def call(self, inputs, **kw):
        PS = self.PS
        seq, typ, idx, val, fit, mlm = inputs
        seq = y = self.transformer([[seq, typ], None], **kw)
        fit_y = self.pool(T.squeeze(y[:, 0:1, :], axis=1), **kw)
        y = T.gather(y, idx, axis=1)
        y = self.norm(self.mlm_dense(y, **kw), **kw)
        y = T.matmul(y, self.embed, transpose_b=True)
        y = T.nn.log_softmax(K.bias_add(y, self.mlm_bias), axis=-1)
        mlm_loss = -K.sum(y * K.one_hot(val, PS.vocab_size), axis=-1)
        y = T.matmul(fit_y, self.gain, transpose_b=True)
        y = T.nn.log_softmax(K.bias_add(y, self.bias), axis=-1)
        fit_loss = -K.sum(y * K.one_hot(fit, 2), axis=-1)
        loss = K.sum(mlm * mlm_loss) / (K.sum(mlm) + 1e-5) + K.mean(fit_loss)
        return seq, loss


class Transformer(KL.Layer):
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
        self.drop = KL.Dropout(PS.hidden_drop)
        pre, post = PreProc(PS), PostProc(PS)
        self.e_stack = EncodeStack(PS, pre, post)
        self.d_stack = DecodeStack(PS, pre, post)
        self.logits = KL.Dense(PS.vocab_size, activation=None)

    def compute_output_shape(self, input_shape):
        e, d = None, None
        src, tgt = input_shape
        if src:
            e, _ = self.e_stack.output_shape
        if tgt:
            d = self.logits.output_shape
        return e, d

    def call(self, inputs, **kw):
        e, a, d = None, None, None
        src, tgt = inputs
        if src:
            x, typ = src
            y = self.tok_embed(x, **kw)
            if self.typ_embed:
                y = self.typ_embed([y, typ], **kw)
            if self.pos_embed:
                y = self.pos_embed(y, **kw)
            y = self.norm(y, **kw)
            y = self.drop(y, **kw)
            e, a = self.e_stack(y, **kw)
        if tgt:
            x, typ = tgt
            y = self.tok_embed(x, **kw)
            if self.typ_embed:
                y = self.typ_embed([y, typ], **kw)
            if self.pos_embed:
                y = self.pos_embed(y, **kw)
            y = self.norm(y, **kw)
            y = self.drop(y, **kw)
            d = self.d_stack([e, a, y], **kw)
            d = self.logits(d, **kw)
        return e, d


"""
    def get_config(self):
        c = super().get_config()
        c['PS'] = self.PS
        return c
"""


class TokEmbed(KL.Embedding):
    def __init__(self, PS, **_):
        ei = _get_initer(PS.init_stddev)
        er = KS.regularizers.l2(PS.l2_penalty) if PS.l2_penalty else None
        super().__init__(
            input_dim=PS.vocab_size + 1,
            output_dim=PS.hidden_size,
            embeddings_initializer=ei,
            embeddings_regularizer=er,
            input_length=PS.max_seq_len,
            mask_zero=True,
        )


class TypEmbed(KL.Layer):
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


class PosEmbed(KL.Layer):
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
        self.bias = T.slice(b, [0, 0], [xlen, -1])
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + K.expand_dims(self.bias, 0)


class PosTiming(KL.Layer):
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
        s = N.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * K.exp(K.arange(n, dtype=K.floatx()) * -s)
        p = K.arange(xlen, dtype=K.floatx()) + self.start
        p = K.expand_dims(p, 1) * K.expand_dims(s, 0)
        p = K.concatenate([K.sin(p), K.cos(p)], axis=1)
        self.bias = K.expand_dims(p, axis=0)
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + self.bias


class LayerNorm(KL.Layer):
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


class Stack(KL.Layer):
    prox_bias = None

    @staticmethod
    def proximity(slen):
        p = K.arange(slen, dtype=K.floatx())
        p = K.expand_dims(p, 0) - K.expand_dims(p, 1)
        return K.expand_dims(K.expand_dims(-T.math.log1p(K.abs(p)), 0), 0)

    @staticmethod
    def attn_bias(mask):
        f = K.floatx()
        fmin = T.float16.min if f == 'float16' else T.float32.min
        b = K.cast(mask, f) * fmin
        return K.expand_dims(K.expand_dims(b, axis=1), axis=1)

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
        return self.encs[-1].output_shape

    def call(self, inputs, mask, **kw):
        x = inputs
        b = self.attn_bias(mask)
        if self.prox_bias:
            b += self.prox_bias
        # if self.PS.pad_remover:
        #     kw.update(pad_remover=U.PadRemover(mask))
        y = self.pre.drop(x, **kw)
        for e in self.encs:
            y = e([y, b], **kw)
        return self.post([x, y], **kw), b


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
        return self.decs[-1].output_shape

    def call(self, inputs, mask, **kw):
        x, b, t = inputs
        sb = self.attn_bias(mask[2])
        PS = self.PS
        if PS.causal_self_attn:
            if PS.prepend_mode == 'prepend_inputs_full_attention':
                p = K.cumsum(K.cumsum(sb, axis=1), axis=1)
                p = K.greater(K.expand_dims(p, 1), K.expand_dims(p, 2))
                sb = K.expand_dims(K.cast(p, K.floatx()) * -1e9, 1)
            else:
                ln = K.int_shape(t)[1]
                sh = (1, 1, ln, ln)
                b = U.ones_band_part(ln, ln, -1, 0, out_shape=sh)
                sb = -1e9 * (1.0 - b)
        if self.prox_bias:
            sb += self.prox_bias
        # y = T.pad(t, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # t = T.concat([pad_value, t], axis=1)[:, :-1, :]
        y = self.pre.drop(t, **kw)
        for d in self.decs:
            y = d([x, b, y, sb], **kw)
        return K.expand_dims(self.post([t, y], **kw), axis=2)


class Encoder(KL.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.reflection = _attentions[PS.self_attn_type](*a)
        self.ffn = _ffns[PS.ffn_layer](*a)

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        x, b = inputs
        y = self.reflection([x, x, b], **kw)
        return self.ffn(y, **kw)


class Decoder(KL.Layer):
    def __init__(self, PS, pre, post, **kw):
        super().__init__(**kw)
        a = (PS, pre, post)
        self.reflection = _attentions[PS.self_attn_type](*a)
        self.attention = _attentions[PS.attn_type](*a)
        self.ffn = _ffns[PS.ffn_layer](*a, conv_pad='LEFT')

    def compute_output_shape(self, _):
        return self.ffn.output_shape

    def call(self, inputs, **kw):
        x, b, t, sb = inputs
        y = self.reflection([t, t, sb], **kw)
        y = self.attention([y, x, b], **kw)
        return self.ffn(y, **kw)


class Attention(KL.Layer):
    def __init__(self, PS, pre, post, comp=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        self.comp = comp or self.dense_comp

    def build(self, input_shape):
        src, tgt, _ = input_shape
        hs = src[2]
        assert hs == tgt[2]
        PS = self.PS
        assert hs == PS.hidden_size
        n = PS.attn_heads
        assert hs % n == 0
        self.q_comp = self.comp(hs, name='Q')
        self.k_size = ks = PS.attn_k_size or hs
        assert ks % n == 0
        self.k_comp = self.comp(ks, name='K')
        vs = PS.attn_v_size or hs
        assert vs % n == 0
        self.v_comp = self.comp(vs, name='V')
        self.out = self.dense_comp(hs, name='out')
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.out.output_shape

    def call(self, inputs, **kw):
        s, t, b = inputs
        s = self.pre(s, **kw)
        q = self.split_heads(self.q_comp(s, **kw))
        k = self.split_heads(self.k_comp(t, **kw))
        v = self.split_heads(self.v_comp(t, **kw))
        y = self.scores(q, k, v, b, **kw)
        y = self.join_heads(y)
        y = self.out(y)
        return self.post([s, y], **kw)

    def dense_comp(self, size, **kw):
        ki = _get_initer(self.PS.init_stddev)
        return KL.Dense(size, use_bias=False, kernel_initializer=ki, **kw)

    def split_heads(self, x):
        sh = K.int_shape(x)
        n = self.PS.attn_heads
        y = K.reshape(x, (-1, sh[1], n, sh[-1] // n))
        return K.permute_dimensions(y, [0, 2, 1, 3])

    def scores(self, q, k, v, b, **kw):
        raise NotImplementedError()

    @staticmethod
    def join_heads(x):
        y = K.permute_dimensions(x, [0, 2, 1, 3])
        sh = K.int_shape(y)
        return K.reshape(y, (-1, sh[1], sh[2] * sh[3]))


class ConvComp(KL.Layer):
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
        self.conv = KL.Conv1D(filters, ksize, **kw)

    def call(self, inputs, **kw):
        x = inputs
        if self.padding == 'LEFT':
            sh = K.int_shape(x)
            # h = 2 * (self.ksize // 2) * self.dilation_rate[0]
            # w = 0 if sh[2] == 1 else 2 * (ks[1] // 2) * self.dilation_rate[1]
            # p = T.constant([[0, 0], [h, 0], [w, 0], [0, 0]])
            # x = T.pad(x, p)
            # x.set_shape([sh[0], None, None, sh[3]])
        return self.conv(x)


class DotAttn(Attention):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.drop = KL.Dropout(self.PS.attn_drop)

    def scores(self, q, k, v, b, **kw):
        y = T.matmul(q, k, transpose_b=True)
        y *= (self.k_size // self.PS.attn_heads)**-0.5
        y = self.drop(KS.activations.softmax(y + b, **kw), **kw)
        return T.matmul(y, v)


_attentions = {
    'dot_attn': DotAttn,
}


class FForward(KL.Layer):
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
        self.dense1 = KL.Dense(PS.ffn_units, activation=ac, **kw)
        self.drop = KL.Dropout(PS.ffn_drop)
        self.dense2 = KL.Dense(PS.hidden_size, **kw)

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


class Processor(KL.Layer):
    cmd = None

    @staticmethod
    def _dropout(rate, shape, bdims):
        ns, bds = None, [int(i) for i in bdims.split(',') if i]
        if bds:
            n = len(shape)
            bds = [d + n if d < 0 else d for d in bds]
            ns = [1 if i in bds else shape[i] for i in range(n)]
        return KL.Dropout(rate, noise_shape=ns)

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.drop = self._dropout(PS.prepost_drop, (), PS.prepost_bdims)
        self.batch = KL.BatchNormalization(epsilon=PS.norm_epsilon)

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
                        m, v = T.nn.moments(x, [1, 2, 4], keep_dims=True)
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
    return KS.initializers.TruncatedNormal(stddev=stddev)


def _gelu(x):
    c = K.tanh((N.sqrt(2 / N.pi) * (x + 0.044715 * K.pow(x, 3))))
    c = (c + 1.0) * 0.5
    return x * c


def _get_act(name):
    if isinstance(name, str):
        n = name.lower()
        if n == 'gelu':
            return _gelu
        elif n == 'relu':
            return KS.activations.relu
        elif n == 'tanh':
            return KS.activations.tanh
        else:
            assert n == 'linear'
            name = None
    return name
