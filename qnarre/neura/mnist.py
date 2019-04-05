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

from datetime import datetime

import tensorflow as tf

from qnarre.neura import utils
from qnarre.neura.params import load_flags, load_params
from qnarre.feeds.dset.mnist_ds import dataset as mnist_ds

ks = tf.keras
kls = ks.layers

# kcb = ks.callbacks


def model_for(params):
    PS = params
    f = PS.data_format
    shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]
    x = y = kls.Input((784, ))
    ls = {
        'mlp': [
            kls.Reshape(shape),
            kls.Flatten(),
            kls.Dense(PS.num_units, activation='relu'),
            kls.Dropout(PS.dropout_rate),
            kls.Dense(PS.num_classes, activation='softmax'),
        ],
        'cnn': [
            kls.Reshape(shape),
            kls.Conv2D(32, 3, activation='relu'),
            kls.Conv2D(64, 3, activation='relu'),
            kls.MaxPooling2D(2),
            kls.Dropout(0.25),
            kls.Flatten(),
            kls.Dense(128, activation='relu'),
            kls.Dropout(0.5),
            kls.Dense(PS.num_classes, activation='softmax'),
        ],
        'cnn_2': [
            kls.Reshape(shape),
            kls.Conv2D(32, 5, padding='same', activation='relu'),
            kls.MaxPooling2D(2, padding='same'),
            kls.Conv2D(64, 5, padding='same', activation='relu'),
            kls.MaxPooling2D(2, padding='same'),
            kls.Flatten(),
            kls.Dense(1024, activation='relu'),
            kls.Dropout(0.4),
            kls.Dense(PS.num_classes),
        ],
        'ds_api': [
            kls.Reshape(shape),
            kls.Conv2D(32, 3, padding='valid', activation='relu'),
            kls.MaxPooling2D(2),
            kls.Conv2D(64, 3, activation='relu'),
            kls.MaxPooling2D(2),
            kls.Flatten(),
            kls.Dense(512, activation='relu'),
            kls.Dropout(0.5),
            kls.Dense(PS.num_classes, activation='softmax'),
        ],
        'hrnn': [
            kls.Reshape(shape),
            kls.TimeDistributed(kls.LSTM(128)),
            kls.LSTM(128),
            kls.Dense(PS.num_classes, activation='softmax'),
        ],
        'irnn': [
            kls.Reshape(shape),
            kls.SimpleRNN(
                10,
                kernel_initializer=ks.initializers.RandomNormal(stddev=0.001),
                recurrent_initializer=ks.initializers.Identity(gain=1.0),
                activation='relu'),
            kls.Dense(PS.num_classes),
            kls.Activation('softmax'),
        ],
    }[PS.model_name]
    for lr in ls:
        y = lr(y)
    m = ks.Model(inputs=[x], outputs=[y])
    m.compile(
        optimizer=PS.optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return m


def dset_for(kind, params):
    PS = params
    ds = mnist_ds(kind, PS)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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

_params.update(
    data_dir='.data/mnist',
    log_dir='.model/mnist/logs',
    model_dir='.model/mnist',
    save_dir='.model/mnist/save',
)


def main(_):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
    PS = load_params().override(_params)
    utils.train_sess(sid, PS, model_for, dset_for)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
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
    preds = np.argmax(model.predict(ds_test), axis=1)
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    img = _to_image(_to_plot(cm, names))
    with writer.as_default():
        tf.summary.image('Confusion Matrix', img, step=epoch)

    cbacks = [
        kcb.LambdaCallback(on_epoch_end=log_confusion_matrix),
    ]
"""
