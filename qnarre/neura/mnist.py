#!/usr/bin/env python
# Copyright 2018 Quantapix Authors. All Rights Reserved.
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

import pathlib as P

from datetime import datetime

import tensorflow as T

from qnarre.neura import utils as U
from qnarre.neura.params import load_flags, load_params
from qnarre.feeds.dset.mnist_ds import dataset as mnist_ds

KS = T.keras
K = KS.backend
KL = KS.layers
KC = KS.callbacks


class Mnist_1(KL.Layer):
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
        self.d1_1 = KL.Dense(hsize, activation='relu')
        self.d1_2 = KL.Dense(hsize, activation='relu')
        self.d2_1 = KL.Dense(10, activation='softmax')
        self.d2_2 = KL.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, yt1, x2, yt2 = inputs[:4]
        y1, y2 = KL.Reshape(self.shape)(x1), KL.Reshape(self.shape)(x2)
        y1, y2 = KL.Flatten()(y1), KL.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = KL.Dropout(0.1)(y1), KL.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        l1 = K.sparse_categorical_crossentropy(
            K.cast(yt1, K.floatx()), y1, from_logits=False, axis=-1)
        l2 = K.sparse_categorical_crossentropy(
            K.cast(yt2, K.floatx()), y2, from_logits=False, axis=-1)
        self.add_loss((l1 + l2) / 2.0)
        return [y1, y2]


class Mnist_2(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        f = PS.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        x1, _, x2 = input_shape[:3]
        _, hsize = x1
        _, hs = x2
        assert hsize == hs
        self.d1_1 = KL.Dense(hsize, activation='relu')
        self.d1_2 = KL.Dense(hsize, activation='relu')
        self.d2_1 = KL.Dense(10, activation='softmax')
        self.d2_2 = KL.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        x1, _, x2 = inputs[:3]
        y1, y2 = KL.Reshape(self.shape)(x1), KL.Reshape(self.shape)(x2)
        y1, y2 = KL.Flatten()(y1), KL.Flatten()(y2)
        y1, y2 = self.d1_1(y1), self.d1_2(y2)
        y1, y2 = KL.Dropout(0.1)(y1), KL.Dropout(0.1)(y2)
        y1, y2 = self.d2_1(y1), self.d2_2(y2)
        return [y1, y2]


class Mnist_3(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        f = PS.data_format
        self.shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]

    def build(self, input_shape):
        x = input_shape[4]
        _, hsize = x
        self.d1 = KL.Dense(hsize, activation='relu')
        self.d2 = KL.Dense(10, activation='softmax')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        PS = self.PS
        x = inputs[4]
        y = KL.Reshape(self.shape)(x)
        y = KL.Flatten()(y)
        y = self.d1(y)
        y = KL.Dropout(0.1)(y)
        y = self.d2(y)
        return y


def model_for(params):
    PS = params
    FS = PS.features
    ins = [
        KL.Input(**FS.input_kw(FS.IMG)),
        KL.Input(**FS.input_kw(FS.LBL)),
        KL.Input(**FS.input_kw(FS.IMG)),
        KL.Input(**FS.input_kw(FS.LBL)),
        KL.Input(**FS.input_kw(FS.IMG)),
        KL.Input(**FS.input_kw(FS.LBL)),
    ]
    print(ins)
    outs = [Mnist_3(PS)(ins[:5])]
    # outs = [Mnist_1(PS)(ins), Mnist_2(PS)(ins), Mnist_3(PS)(ins)]
    m = KS.Model(inputs=ins[:5], outputs=outs)
    m.compile(
        optimizer=KS.optimizers.SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        target_tensors=[ins[5]],
    )
    """
        loss={
            'mnist_1': None,
            'mnist_2': 'sparse_categorical_crossentropy',
            'mnist_3': 'sparse_categorical_crossentropy',
        },
        target_tensors={
            'mnist_1': None,
            'mnist_2': ['input_2', 'input_4'],
            'mnist_3': ['input_6'],
        },
    """
    return m


def dset_for(params, kind):
    PS = params
    ds_1 = mnist_ds(PS, kind)
    ds_2 = mnist_ds(PS, kind)
    ds_3 = mnist_ds(PS, kind)
    if kind == 'train':
        ds_1 = ds_1.shuffle(buffer_size=50000)
        ds_2 = ds_2.shuffle(buffer_size=50000)
        ds_3 = ds_3.shuffle(buffer_size=50000)
    ds = T.data.Dataset.zip((ds_1, ds_2, ds_3))
    ds = ds.map(lambda *ts: (tuple([v for t in ts for v in t]), ))
    ds = ds.batch(PS.batch_size)
    ds = ds.prefetch(buffer_size=T.data.experimental.AUTOTUNE)
    print(ds)
    return ds


_params = dict(
    batch_size=64,
    dropout_rate=0.2,
    epochs_between_evals=1,
    model_name='mlp',
    num_classes=10,
    num_units=512,
    optimizer='adam',
    train_epochs=2,
)

_fspecs = {
    'IMG': {
        'name': 'image',
        'dtype': 'float32',
        'shape': (28 * 28, ),
    },
    'LBL': {
        'name': 'label',
        'dtype': 'float32',
        'shape': (),
    },
}

_params.update(
    features=U.Features(specs=_fspecs),
    data_dir='.data/mnist',
    log_dir='.model/mnist/logs',
    model_dir='.model/mnist',
    save_dir='.model/mnist/save',
)


def main(_):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
    PS = load_params().override(_params)
    train_sess(sid, PS, model_for, dset_for)


def train_sess(sid, params, model_fn, dset_fn, cbacks=None):
    PS = params
    # with T.distribute.MirroredStrategy().scope():
    m = model_fn(PS)
    ds_train = dset_fn(PS, 'train')
    ds_test = dset_fn(PS, 'test')
    m.summary()
    hist = m.fit(ds_train, epochs=PS.train_epochs, validation_data=ds_test)
    print(f'History: {hist.history}')
    loss, acc = m.evaluate(ds_test)
    print(f'\nTest loss, acc: {loss}, {acc}')


def train_sess2(sid, params, model_fn, dset_fn, cbacks=None):
    PS = params
    # with T.distribute.MirroredStrategy().scope():
    m = model_fn(PS)
    ds_train = dset_fn(PS, 'train')
    ds_test = dset_fn(PS, 'test')
    loss_obj = KS.losses.SparseCategoricalCrossentropy()
    opt = KS.optimizers.SGD(learning_rate=0.01)

    for epoch in range(3):
        print('Start of epoch %d' % (epoch, ))
        for step, (x, ) in enumerate(ds_train):
            x1, y1, x2, y2, x3, y3 = x
            with T.GradientTape() as tape:
                logits = m(x)
                loss = loss_obj(y3, logits)
                loss += sum(m.losses)
            grads = tape.gradient(loss, m.trainable_variables)
            opt.apply_gradients(zip(grads, m.trainable_variables))


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    load_flags()
    from absl import flags
    flags.DEFINE_float('dropout_rate', None, '')
    flags.DEFINE_integer('num_classes', None, '')
    flags.DEFINE_integer('num_units', None, '')
    flags.DEFINE_string('optimizer', None, '')
    from absl import app
    app.run(main)

###
"""
names = [str(i) for i in range(PS.num_classes)]
labels = [lb.numpy() for _, lb in ds_test]

def log_confusion_matrix(epoch, logs):
    preds = N.argmax(model.predict(ds_test), axis=1)
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    img = _to_image(_to_plot(cm, names))
    with writer.as_default():
        T.summary.image('Confusion Matrix', img, step=epoch)

    cbacks = [
        kcb.LambdaCallback(on_epoch_end=log_confusion_matrix),
    ]
"""
