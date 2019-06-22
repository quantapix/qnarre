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


class Norm(kl.Layer):
    def build(self, shape):
        s = shape[-1]
        self.n_w = self.add_weight(name='n_w', shape=s, initializer='ones')
        self.n_b = self.add_weight(name='n_b', shape=s, initializer='zeros')
        return super().build(shape)

    @tf.function
    def call(self, x, mask=None):
        if mask is not None:
            x *= tf.cast(mask, tf.float32)[:, :, None]
        m = tf.reduce_mean(x, axis=-1, keepdims=True)
        v = tf.reduce_mean(tf.square(x - m), axis=-1, keepdims=True)
        y = (x - m) / tf.sqrt(v + 1e-6)
        y = y * self.n_w + self.n_b
        return y


params = dict(
    dim_hidden=1000,
    dim_input=100,
    loss=ks.losses.MeanAbsoluteError,
    metrics=ks.metrics.MeanAbsoluteError,
    num_layers=10,
    optimizer=ks.optimizers.SGD,
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def main(_):
    ps = Params(**params)
    d = np.ones((100, ps.dim_input))
    # with tf.distribute.MirroredStrategy().scope():
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(d, d, callbacks=cs, epochs=10, batch_size=10)


if __name__ == '__main__':
    from absl import app
    app.run(main)
