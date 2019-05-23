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

import pathlib as pth

from datetime import datetime

import tensorflow as tf

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams

from qfeeds.roc.dataset import dataset as roc_ds

import qneura.quess.utils as qu

from qneura.quess.layers import Quess
from qneura.quess.params import Params, load_flags

ks = tf.keras
K = ks.backend
kls = ks.layers
kcb = ks.callbacks

opt_fn = qu.adam_opt
loss_fn = qu.xent_loss


def dataset_for(kind, params):
    PS = params
    ds = roc_ds(kind, PS)
    if kind == 'train':
        ds = ds.repeat().shuffle(PS.shuffle_size)
    else:
        assert kind == 'test'
    sh = (PS.src_len or PS.mem_len, PS.tgt_len or PS.mem_len, 1)
    ds = ds.padded_batch(PS.batch_size, sh)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def model_for(params):
    PS = params
    sl = PS.src_len or PS.mem_len
    s = kls.Input(shape=(sl, ), dtype='int32', name='source')
    tl = PS.tgt_len or PS.mem_len
    t = kls.Input(shape=(tl, ), dtype='int32', name='target')
    o, y = Quess(PS)([s, t])
    y = kls.Softmax(name='preds')(y)
    m = ks.Model(inputs=[s, t], outputs=[y])
    # m.add_loss(Q.reduce_mean(PS.penalty * K.sum(y * K.log(y), axis=-1)))
    # m.add_loss(tf.reduce_sum(x1) * 0.1)
    # m.add_metric(K.std(x1), name='std_of_activation', aggregation='mean')
    return m


def compile_args_for(params):
    PS = params
    return dict(
        optimizer=opt_fn(PS),
        loss=loss_fn,
        metrics=[ks.metrics.sparse_categorical_accuracy])


def run_quess(sess, params):
    PS = params

    ds_train = dataset_for('train', PS)
    ds_test = dataset_for('test', PS)

    # with tf.distribute.MirroredStrategy().scope():
    model = model_for(PS)
    model.compile(**compile_args_for(PS))
    model.train_on_batch(ds_train[:1])
    save_p = pth.Path(PS.dir_save)
    if save_p.exists():
        model.load_weights(save_p)
    model.summary()

    p = PS.log_dir + '/train/' + sess
    writer = tf.summary.create_file_writer(p)
    sum_s = hparams.session_start_pb(hparams=PS.hparams)

    cbacks = [
        # kcb.LambdaCallback(on_epoch_end=log_confusion_matrix),
        kcb.History(),
        kcb.BaseLogger(),
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
        epochs=PS.train_steps // PS.eval_frequency,
        steps_per_epoch=PS.eval_frequency,
        validation_data=ds_test,
        validation_steps=PS.eval_steps,
        callbacks=cbacks,
    )
    print(f'History: {hist.history}')

    if save_p.exists():
        model.save_weights(save_p, save_format='tf')

    loss, acc = model.evaluate(ds_test)
    print(f'\nTest loss, acc: {loss}, {acc}')

    with writer.as_default():
        e = tf.compat.v1.Event(summary=sum_s).SerializeToString()
        tf.summary.import_event(e)
        tf.summary.scalar('accuracy', acc, step=1, description='Accuracy')
        sum_e = hparams.session_end_pb(api_pb2.STATUS_SUCCESS)
        e = tf.compat.v1.Event(summary=sum_e).SerializeToString()
        tf.summary.import_event(e)


def eager_quess(sess, params):
    PS = params

    ds_train = dataset_for('train', PS)
    ds_test = dataset_for('test', PS)

    # with tf.distribute.MirroredStrategy().scope():
    model = model_for(PS)
    model.compile(**compile_args_for(PS))
    # model.train_on_batch(ds_train[:1])
    save_p = pth.Path(PS.dir_save)
    if save_p.exists():
        model.load_weights(save_p)
    model.summary()

    p = PS.log_dir + '/train/' + sess
    writer = tf.summary.create_file_writer(p)
    sum_s = hparams.session_start_pb(hparams=PS.hparams)

    cbacks = [
        # kcb.LambdaCallback(on_epoch_end=log_confusion_matrix),
        kcb.History(),
        kcb.BaseLogger(),
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

    opt = opt_fn(PS)
    for e in range(3):
        print(f'Start of epoch {e}')
        for s, (src, tgt, agree) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                r = model([src, tgt])
                loss = loss_fn(agree, r)
                # loss += sum(model.losses)
                gs = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(gs, model.trainable_variables))
                # acc_metric(fit, f)
            if s % 200 == 0:
                print(f'Loss at step {s}: {loss}')
        # a = acc_metric.result()
        # acc_metric.reset_states()
        # print(f'Train acc over epoch: {float(a)}')

        for src, tgt, agree in ds_test:
            r = model([src, tgt])
            # acc_metric(fit, f)
        # a = acc_metric.result()
        # acc_metric.reset_states()
        # print(f'Test acc: {float(a)}')

    if save_p.exists():
        model.save_weights(save_p, save_format='tf')

    with writer.as_default():
        e = tf.compat.v1.Event(summary=sum_s).SerializeToString()
        tf.summary.import_event(e)
        # tf.summary.scalar('accuracy', acc, step=1, description='Accuracy')
        sum_e = hparams.session_end_pb(api_pb2.STATUS_SUCCESS)
        e = tf.compat.v1.Event(summary=sum_e).SerializeToString()
        tf.summary.import_event(e)


def main(_):
    from absl import flags
    fs = flags.FLAGS
    # print(fs)
    f = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
    ps = Params(flags=fs, data_format=fs.data_format or f)
    s = datetime.now().strftime('%Y%m%d-%H%M%S')
    print(f'--- Running session {s}:')
    # run_quess(s, ps)
    eager_quess(s, ps)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    load_flags()
    from absl import app
    app.run(main)
