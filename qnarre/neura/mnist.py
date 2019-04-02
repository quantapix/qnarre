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

import io
import itertools

import pathlib as pth

from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sklearn.metrics

from google.protobuf import struct_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams

from qfeeds.mnist.dataset import dataset as mnist_ds

ks = tf.keras
kls = ks.layers
kcb = ks.callbacks


def model_for(params):
    f = params.data_format
    shape = [1, 28, 28] if f == 'channels_first' else [28, 28, 1]
    x = y = kls.Input((784, ))
    ls = {
        'mlp': [
            kls.Reshape(shape),
            kls.Flatten(),
            kls.Dense(params.num_units, activation='relu'),
            kls.Dropout(params.dropout_rate),
            kls.Dense(params.num_classes, activation='softmax'),
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
            kls.Dense(params.num_classes, activation='softmax'),
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
            kls.Dense(params.num_classes),
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
            kls.Dense(params.num_classes, activation='softmax'),
        ],
        'hrnn': [
            kls.Reshape(shape),
            kls.TimeDistributed(kls.LSTM(128)),
            kls.LSTM(128),
            kls.Dense(params.num_classes, activation='softmax'),
        ],
        'irnn': [
            kls.Reshape(shape),
            kls.SimpleRNN(
                10,
                kernel_initializer=ks.initializers.RandomNormal(stddev=0.001),
                recurrent_initializer=ks.initializers.Identity(gain=1.0),
                activation='relu'),
            kls.Dense(params.num_classes),
            kls.Activation('softmax'),
        ],
    }[params.model_name]
    for lr in ls:
        y = lr(y)
    return ks.Model(inputs=[x], outputs=[y])


def dataset_for(kind, params):
    ds = mnist_ds(kind, params)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(params.batch_size)
        # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds
    return ds


def run_mnist(sess, params):
    names = [str(i) for i in range(params.num_classes)]
    # with tf.distribute.MirroredStrategy().scope():
    model = model_for(params)
    model.compile(
        optimizer=params.optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    ds_train = dataset_for('train', params)
    ds_test = dataset_for('test', params)
    labels = [lb.numpy() for _, lb in ds_test]
    ds_test = ds_test.batch(params.batch_size)

    save_p = pth.Path(params.save_dir)
    if save_p.exists():
        model.train_on_batch(ds_train[:1])
        model.load_weights(save_p)

    model.summary()

    p = params.log_dir + '/train/' + sess
    writer = tf.summary.create_file_writer(p)
    sum_s = hparams.session_start_pb(hparams=params.hparams)

    def log_confusion_matrix(epoch, logs):
        preds = np.argmax(model.predict(ds_test), axis=1)
        cm = sklearn.metrics.confusion_matrix(labels, preds)
        img = _to_image(_to_plot(cm, names))
        with writer.as_default():
            tf.summary.image("Confusion Matrix", img, step=epoch)

    cbacks = [
        kcb.LambdaCallback(on_epoch_end=log_confusion_matrix),
        kcb.TensorBoard(
            log_dir=p,
            histogram_freq=1,
            embeddings_freq=0,
            update_freq='epoch'),
        # kcb.EarlyStopping(
        #     monitor='val_loss', min_delta=1e-2, patience=2, verbose=True),
    ]

    if save_p.exists():
        cbacks.append(
            kcb.ModelCheckpoint(
                model_save_path=save_p,
                save_best_only=True,
                monitor='val_loss',
                verbose=True))

    hist = model.fit(
        ds_train,
        callbacks=cbacks,
        epochs=params.train_epochs,
        validation_data=ds_test)
    print(f'History: {hist.history}')

    if save_p.exists():
        model.save_weights(save_p, save_format='tf')

    loss, acc = model.evaluate(ds_test)
    print(f'\nTest loss, acc: {loss}, {acc}')

    with writer.as_default():
        e = tf.compat.v1.Event(summary=sum_s).SerializeToString()
        tf.summary.import_event(e)
        tf.summary.scalar('accuracy', acc, step=1, description="Accuracy")
        sum_e = hparams.session_end_pb(api_pb2.STATUS_SUCCESS)
        e = tf.compat.v1.Event(summary=sum_e).SerializeToString()
        tf.summary.import_event(e)


def _to_plot(cm, names):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45)
    plt.yticks(ticks, names)
    cm = np.around(
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def _to_image(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)
    return img


class Params:
    def __init__(self, *, flags, **kw):
        ps = defaultdict(
            lambda: None,
            model_name=None,
            data_dir=None,
            model_dir=None,
            save_dir=None,
            log_dir=None,
            batch_size=None,
            train_epochs=None,
            num_classes=10,
            epochs_between_evals=1,
            num_units=512,
            dropout_rate=0.2,
            optimizer='adam',
        )
        ps.update(**kw)
        fs = {}
        for k, v in ps.items():
            if hasattr(flags, k):
                v = getattr(flags, k)
                if v:
                    fs[k] = v
        ps.update(**fs)
        self.update(**ps)

    @property
    def hparams(self):
        return {
            'num_units': self.num_units,
            'dropout_rate': self.dropout_rate,
            'optimizer': self.optimizer
        }

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def load_flags():
    from official.utils.flags import core as fu
    fu.define_base()
    fu.define_performance(num_parallel_calls=False)
    fu.define_image()
    from absl import flags
    flags.adopt_module_key_flags(fu)
    flags.DEFINE_string('save_dir', None, 'Save dir')
    flags.DEFINE_string('model_name', 'mlp', 'Model name')
    flags.DEFINE_integer('num_classes', 10, 'Number of classes')
    fu.set_defaults(
        data_dir='.data/mnist',
        model_dir='.model/mnist',
        log_dir='.model/mnist/logs',
        save_dir='.model/mnist/save',
        batch_size=64,
        train_epochs=2)


def main(_):
    from absl import flags
    fs = flags.FLAGS
    # print(fs)
    f = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
    ps = Params(flags=fs, data_format=fs.data_format or f)
    nus = [16, 32, 512]
    drs = [0.1, 0.2]
    opts = ['adam', 'sgd']
    writer = tf.summary.create_file_writer(ps.log_dir + '/train')
    with writer.as_default():
        s = _to_summary_pb(nus, drs, opts)
        e = tf.compat.v1.Event(summary=s).SerializeToString()
        tf.summary.import_event(e)
    for nu in nus:
        for dr in drs:
            for opt in opts:
                kw = {'num_units': nu, 'dropout_rate': dr, 'optimizer': opt}
                sess = datetime.now().strftime('%Y%m%d-%H%M%S')
                print(f'--- Running session {sess}:', kw)
                ps.update(**kw)
                run_mnist(sess, ps)


def _to_summary_pb(num_units_list, dropout_rate_list, optimizer_list):
    nus_val = struct_pb2.ListValue()
    nus_val.extend(num_units_list)
    drs_val = struct_pb2.ListValue()
    drs_val.extend(dropout_rate_list)
    opts_val = struct_pb2.ListValue()
    opts_val.extend(optimizer_list)
    return hparams.experiment_pb(
        hparam_infos=[
            api_pb2.HParamInfo(
                name='num_units',
                display_name='Number of units',
                type=api_pb2.DATA_TYPE_FLOAT64,
                domain_discrete=nus_val),
            api_pb2.HParamInfo(
                name='dropout_rate',
                display_name='Dropout rate',
                type=api_pb2.DATA_TYPE_FLOAT64,
                domain_discrete=drs_val),
            api_pb2.HParamInfo(
                name='optimizer',
                display_name='Optimizer',
                type=api_pb2.DATA_TYPE_STRING,
                domain_discrete=opts_val)
        ],
        metric_infos=[
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag='accuracy'),
                display_name='Accuracy'),
        ])


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    load_flags()
    from absl import app
    app.run(main)
