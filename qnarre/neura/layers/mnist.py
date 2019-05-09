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

import qnarre.neura as Q


class Mnist(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        f = PS.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        PS = self.PS
        self.d1 = Q.Dense(PS.hidden_size, activation=PS.hidden_act)
        self.d2 = Q.Dense(PS.num_classes, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x = inputs[0]
        y = Q.Reshape(self.shape)(x)
        y = Q.Flatten()(y)
        y = self.d1(y)
        y = Q.Dropout(self.PS.hidden_drop)(y)
        y = self.d2(y)
        return y


class Mnist_2(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        f = PS.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        print(input_shape)
        x1, _, x2 = input_shape[:3]
        _, hsize = x1
        _, hs = x2
        assert hsize == hs
        self.d1_1 = Q.Dense(hsize, activation='relu')
        self.d1_2 = Q.Dense(hsize, activation='relu')
        self.d2_1 = Q.Dense(10, activation='softmax')
        self.d2_2 = Q.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, yt1, x2, yt2 = inputs[:4]
        y1, y2 = Q.Reshape(self.shape)(x1), Q.Reshape(self.shape)(x2)
        y1, y2 = Q.Flatten()(y1), Q.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = Q.Dropout(0.1)(y1), Q.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        l1 = Q.SparseCategoricalAccuracy(Q.cast(yt1, Q.floatx()),
                                         y1,
                                         from_logits=False,
                                         axis=-1)
        l2 = Q.SparseCategoricalAccuracy(Q.cast(yt2, Q.floatx()),
                                         y2,
                                         from_logits=False,
                                         axis=-1)
        self.add_loss((l1 + l2) / 2.0)
        return [y1, y2]


class Mnist_3(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        f = PS.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        x1, _, x2 = input_shape[:3]
        _, hsize = x1
        _, hs = x2
        assert hsize == hs
        self.d1_1 = Q.Dense(hsize, activation='relu')
        self.d1_2 = Q.Dense(hsize, activation='relu')
        self.d2_1 = Q.Dense(10, activation='softmax')
        self.d2_2 = Q.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, _, x2 = inputs[:3]
        y1, y2 = Q.Reshape(self.shape)(x1), Q.Reshape(self.shape)(x2)
        y1, y2 = Q.Flatten()(y1), Q.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = Q.Dropout(0.1)(y1), Q.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        return [y1, y2]
