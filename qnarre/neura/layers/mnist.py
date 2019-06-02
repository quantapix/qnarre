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

from qnarre.neura import tf
from qnarre.neura.layers import base


def adapter(ps, feats, x):
    d = tf.parse_example(x, feats)
    img = tf.to_dense(d['flt_img'])
    # img = tf.cast(d['int_img'], tf.float32) / 255.
    lbl = d['int_lbl']
    return img, lbl


def model(ps):
    w, h = ps.img_width, ps.img_height
    ins = tf.Input(shape=(w * h, ), dtype='float32'),
    outs = [Mnist(ps)(ins)]
    m = tf.Model(name='MnistModel', inputs=ins, outputs=outs)
    return m


def model_old(ps):
    w, h = ps.img_width, ps.img_height
    ins = [
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(1, ), dtype='int32'),
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(1, ), dtype='int32'),
    ]
    outs = [Mnist(ps)(ins)]
    m = tf.Model(name='MnistModel', inputs=ins, outputs=outs)
    return m


class Mnist(base.Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'data_format',
                'act_hidden',
                'drop_hidden',
                'dim_hidden',
                'img_height',
                'img_width',
                'num_classes',
            ))

    def __init__(self, ps, name=None, **kw):
        super().__init__(ps, name=name or 'Mnist', **kw)
        cfg = self.cfg
        f = cfg.data_format
        self.shape = (1, 28, 28) if f == 'channels_first' else (28, 28, 1)
        self.reshape = tf.Reshape(self.shape)
        self.flatten = tf.Flatten()

    def build(self, input_shape):
        cfg = self.cfg
        self.d1 = tf.Dense(cfg.dim_hidden, activation=cfg.act_hidden, name='d1')
        self.d2 = tf.Dense(cfg.num_classes, activation='softmax', name='d2')
        self.drop = tf.Dropout(cfg.drop_hidden, name='drop')
        return super().build(input_shape)

    @tf.function
    def call(self, inputs):
        x = inputs
        y = self.reshape(x)
        y = self.flatten(y)
        y = self.d1(y)
        y = self.drop(y)
        y = self.d2(y)
        return y


"""
class Mnist_1(base.Layer):
    @staticmethod
    def cfg_items(params):
        return dict(
            params.cfg_items(
                'data_format',
                'act_hidden',
                'drop_hidden',
                'dim_hidden',
                'img_height',
                'img_width',
                'num_classes',
            ))

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        cfg = self.cfg
        f = cfg.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        cfg = self.cfg
        self.d1 = tf.Dense(cfg.dim_hidden, activation=cfg.act_hidden)
        self.d2 = tf.Dense(cfg.num_classes, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x = inputs[0]
        y = tf.Reshape(self.shape)(x)
        y = tf.Flatten()(y)
        y = self.d1(y)
        y = tf.Dropout(self.cfg.drop_hidden)(y)
        y = self.d2(y)
        return y


class Mnist_2(base.Layer):
    def __init__(self, cfg, **kw):
        super().__init__(dtype='float32', **kw)
        f = cfg.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        print(input_shape)
        x1, _, x2 = input_shape[:3]
        _, hsize = x1
        _, hs = x2
        assert hsize == hs
        self.d1_1 = tf.Dense(hsize, activation='relu')
        self.d1_2 = tf.Dense(hsize, activation='relu')
        self.d2_1 = tf.Dense(10, activation='softmax')
        self.d2_2 = tf.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, yt1, x2, yt2 = inputs[:4]
        y1, y2 = tf.Reshape(self.shape)(x1), tf.Reshape(self.shape)(x2)
        y1, y2 = tf.Flatten()(y1), tf.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = tf.Dropout(0.1)(y1), tf.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        l1 = tf.SparseCategoricalAccuracy(tf.cast(yt1, tf.floatx()),
                                          y1,
                                          from_logits=False,
                                          axis=-1)
        l2 = tf.SparseCategoricalAccuracy(tf.cast(yt2, tf.floatx()),
                                          y2,
                                          from_logits=False,
                                          axis=-1)
        self.add_loss((l1 + l2) / 2.0)
        return [y1, y2]


class Mnist_3(base.Layer):
    def __init__(self, cfg, **kw):
        super().__init__(dtype='float32', **kw)
        f = cfg.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        x1, _, x2 = input_shape[:3]
        _, hsize = x1
        _, hs = x2
        assert hsize == hs
        self.d1_1 = tf.Dense(hsize, activation='relu')
        self.d1_2 = tf.Dense(hsize, activation='relu')
        self.d2_1 = tf.Dense(10, activation='softmax')
        self.d2_2 = tf.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, _, x2 = inputs[:3]
        y1, y2 = tf.Reshape(self.shape)(x1), tf.Reshape(self.shape)(x2)
        y1, y2 = tf.Flatten()(y1), tf.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = tf.Dropout(0.1)(y1), tf.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        return [y1, y2]
"""
