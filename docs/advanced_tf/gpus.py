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
# !pip install tensorflow==2.0.0-beta0

import numpy as np
import tensorflow as tf

from datetime import datetime

keras = tf.keras
layers = tf.keras.layers


class Layer(layers.Layer):
    def __init__(self, ps, **kw):
        super().__init__(**kw)
        self.ps = ps

    def build(self, input_shape):
        s = input_shape[-1]
        self.w = self.add_weight(name='l_w', shape=(s, s))
        self.b = self.add_weight(name='l_b', shape=(s, ))
        return super().build(input_shape)

    def call(self, x):
        y = tf.einsum('bi,ij->bj', x, self.w) + self.b
        return y


def model_for(ps):
    m = keras.Sequential()
    m.add(layers.Dense(ps.dim_hidden, input_dim=ps.dim_input, name='in'))
    for i in (range(ps.num_layers)):
        m.add(Layer(ps, name=f'lay_{i}'))
    m.add(layers.Dense(ps.dim_input, name='out'))
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metrics])
    print(m.summary())
    return m


params = dict(
    dim_hidden=1000,
    dim_input=100,
    loss=keras.losses.MeanAbsoluteError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
    num_layers=10,
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main(_):
    ps = Params(**params)
    # with T.distribute.MirroredStrategy().scope():
    d = np.ones((100, ps.dim_input))
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/logs/{ld}'
    cs = [keras.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(d, d, callbacks=cs, epochs=10, batch_size=10)


if __name__ == '__main__':
    from absl import app
    app.run(main)
