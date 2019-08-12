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

from datetime import datetime

import numpy as np
import tensorflow as tf

ks = tf.keras
kl = ks.layers
cfg = tf.config.experimental

# tf.debugging.set_log_device_placement(True)

devs = ((None, None, None, None, None), )
devs = ((None, ), (1000, 1000, 1000, 1000, 1000, 1000),
        (1000, 1000, 1000, 1000, 1000, 1000))
cfg.set_visible_devices(cfg.get_visible_devices('CPU')[:1], 'CPU')
cfg.set_visible_devices(cfg.get_visible_devices('GPU')[:len(devs) - 1], 'GPU')
for d, ms in zip(cfg.get_visible_devices(), devs):
    vs = [cfg.VirtualDeviceConfiguration(m) for m in ms]
    cfg.set_virtual_device_configuration(d, vs)
devs = cfg.list_logical_devices('CPU')
devs += cfg.list_logical_devices('GPU')
print('devices:', [d.name for d in devs])

tf.config.set_soft_device_placement(False)
# cfg.set_device_policy('warn')


class Layer(kl.Layer):
    def __init__(self, i, ps, **kw):
        super().__init__(**kw)
        self.idx = min(i + 1, len(devs) - 1)
        self.ps = ps

    def build(self, input_shape):
        s = input_shape[-1]
        with tf.device(devs[self.idx].name):
            self.w = self.add_weight(name='l_w', shape=(s, s))
            self.b = self.add_weight(name='l_b', shape=(s, ))
        return super().build(input_shape)

    def call(self, x):
        with tf.device(devs[self.idx].name):
            y = tf.einsum('bi,ij->bj', x, self.w) + self.b
        return y


def model_for(ps):
    m = ks.Sequential()
    m.add(kl.Dense(ps.dim_hidden, input_dim=ps.dim_input, name='in'))
    for i in range(ps.num_layers):
        m.add(Layer(i, ps, name=f'lay_{i}'))
    m.add(kl.Dense(ps.dim_input, name='out'))
    m.compile(optimizer=ps.optimizer(), loss=ps.loss(), metrics=[ps.metrics()])
    print(m.summary())
    return m


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
    ds = np.ones((100, ps.dim_input))  #pylint: disable=no-member
    # with tf.distribute.MirroredStrategy().scope():
    m = model_for(ps)
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, ds, callbacks=cs, epochs=10, batch_size=10)


if __name__ == '__main__':
    from absl import app
    app.run(main)
