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
from qnarre.neura.layers import Squad, SquadLoss
from qnarre.feeds.dset.trafo_ds import dataset as trafo_ds

KS = T.keras


def dataset_for(params, kind):
    PS = params
    ds = trafo_ds(PS, kind)
    n = 100
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(PS.batch_size)
    return ds


def model_for(params):
    PS = params
    FS = PS.features
    seq = KS.Input(**FS.input_kw(FS.SEQ))
    typ = KS.Input(**FS.input_kw(FS.TYP))
    opt = KS.Input(**FS.input_kw(FS.OPT))
    beg = KS.Input(**FS.input_kw(FS.BEG))
    end = KS.Input(**FS.input_kw(FS.END))
    uid = KS.Input(**FS.input_kw(FS.UID))
    ins = [seq, typ, opt, beg, end, uid]
    y = Squad(PS)([seq, typ])
    y = SquadLoss(PS)([beg, end], y)
    m = KS.Model(inputs=ins, outputs=[y])
    m.build()
    # m.compile(optimizer=optimizer_for(PS),
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])
    return m


def optimizer_for(params):
    PS = params
    return KS.optimizers.Adam(learning_rate=PS.adam_beta1,
                              beta_1=PS.adam_beta1,
                              beta_2=PS.adam_beta2,
                              epsilon=PS.adam_epsilon)


_params = dict(
    batch_size=4,
    learn_rate=5e-6,
    src_len=16,
    tgt_len=0,
    vocab_size=0,
    adam_lr=0.001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-7,
)

_params.update(
    data_dir='.data/trafo',
    log_dir='.model/trafo/logs',
    model_dir='.model/trafo',
    save_dir='.model/trafo/save',
)

_fspecs = {
    'SRC': {
        'name': 'source',
        'dtype': 'int32',
        'shape': (None, ),
    },
    'TYP': {
        'name': 'types',
        'dtype': 'int32',
        'shape': (None, ),
    },
    'TGT': {
        'name': 'target',
        'dtype': 'int32',
        'shape': (None, ),
    },
}


class Features(U.Features):
    def __init__(self, params, **kw):
        super().__init__(**kw)
        PS = params
        sh = (PS.src_len, )
        self.shapes[self.SRC] = sh
        self.shapes[self.TYP] = sh
        sh = (PS.tgt_len, )
        self.shapes[self.TGT] = sh


def main(_):
    PS = U.Params(_params)
    PS.update(features=Features(PS, specs=_fspecs))

    model = model_for(PS)
    optimizer = optimizer_for(PS)
    losses = KS.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = KS.metrics.SparseCategoricalAccuracy()

    def train_step(x, y):
        with T.GradientTape() as tape:
            logits = model(x)
            loss = losses(y, logits)
            acc = metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    @T.function
    def train():
        step, loss, acc = 0, 0.0, 0.0
        for x, y in dataset_for(PS, 'train'):
            step += 1
            loss, acc = train_step(x, y)
            if T.equal(step % 10, 0):
                m = metrics.result()
                T.print('Step:', step, ', loss:', loss, ', acc:', m)
        return step, loss, acc

    step, loss, acc = train()
    print('Final step:', step, ', loss:', loss, ', acc:', metrics.result())


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('src_len', None, '')
    from absl import app
    app.run(main)
