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

import tensorflow as T

from qnarre.neura import utils as U
# from qnarre.neura.layers import Trafo
from qnarre.feeds.dset.trafo_ds import dataset as trafo_ds

KS = T.keras
K = KS.backend
KL = KS.layers


class Trafo(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.tok_embed = TokEmbed(PS)
        self.enc_stack = KL.Dense(2 * PS.hidden_size, activation='relu')
        self.dec_stack = KL.Dense(PS.hidden_size, activation='relu')
        self.logits = KL.Dense(PS.vocab_size, activation=None)

    def build(self, input_shape):
        ctx, _, tgt = input_shape
        return super().build(input_shape)

    def call(self, inputs, training=None, **kw):
        ctx, _, tgt = inputs
        y = self.tok_embed(ctx, **kw)
        y = self.enc_stack(y, **kw)
        y = self.dec_stack(y, **kw)
        if training:
            print('training...')
        return self.to_logits(y, **kw)

    def to_logits(self, x, unks=None, prior=None, **kw):
        xs = K.int_shape(x)
        y = K.reshape(x, (-1, xs[-1]))
        y = self.logits(y, **kw)
        ys = K.int_shape(y)
        y = K.reshape(y, (-1, ) + xs[1:-1] + ys[-1:])
        if unks:
            y = T.where(unks, y, prior)
        return y


def dataset_for(params, kind):
    PS = params
    ds = trafo_ds(PS, kind)
    n = 1000
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(PS.batch_size)
    return ds


def model_for(params):
    PS = params
    ctx = KS.Input(shape=(PS.ctx_len, ), dtype='int32')
    typ = KS.Input(shape=(PS.ctx_len, ), dtype='int32')
    tgt = KS.Input(shape=(PS.tgt_len, ), dtype='int32')
    ins = [ctx, typ, tgt]
    y = Trafo(PS)(ins)
    m = KS.Model(inputs=ins, outputs=[y])
    # m.build()
    # m.compile(optimizer=optimizer_for(PS),
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])
    return m


_params = dict(
    batch_size=4,
    hidden_size=8,
    learn_rate=5e-6,
    ctx_len=16,
    tgt_len=0,
    max_pos=0,
    vocab_size=20,
    token_types=8,
)

_params.update(
    data_dir='.data/trafo',
    log_dir='.model/trafo/logs',
    model_dir='.model/trafo',
    save_dir='.model/trafo/save',
)


def main(_):
    PS = U.Params(_params).init_comps()
    model = model_for(PS)

    @T.function
    def train_step(x, y):
        with T.GradientTape() as tape:
            logits = model(x)
            loss = PS.losses(y, logits)
            acc = PS.metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        PS.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    def train():
        step, loss, acc = 0, 0.0, 0.0
        for x, y in dataset_for(PS, 'train'):
            step += 1
            loss, acc = train_step(x, y)
            if T.equal(step % 10, 0):
                m = PS.metrics.result()
                T.print('Step:', step, ', loss:', loss, ', acc:', m)
        return step, loss, acc

    step, loss, acc = train()
    print('Final step:', step, ', loss:', loss, ', acc:', PS.metrics.result())


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('src_len', None, '')
    from absl import app
    app.run(main)
