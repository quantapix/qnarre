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

from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sklearn.metrics

from google.protobuf import struct_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams

ks = tf.keras
K = ks.backend
kls = ks.layers
kcb = ks.callbacks


def adam_opt(params):
    PS = params
    return ks.Adam(
        learning_rate=LRSchedule(),
        beta_1=PS.adam_beta1,
        beta_2=PS.adam_beta2,
        epsilon=PS.adam_epsilon)


def xent_loss(y, x):
    return ks.backend.sparse_categorical_crossentropy(y, x, from_logits=True)


class LRSchedule(ks.optimizers.schedules.LearningRateSchedule):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.constant = PS.lr_constant
        self.schedule = PS.lr_schedule
        self.warmup_steps = PS.lr_warmup_steps

    def __call__(self, step):
        lr = tf.constant(1.0)
        for name in [n.strip() for n in self.schedule.split('*')]:
            if name == 'constant':
                lr *= self.constant
            elif name == 'linear_warmup':
                lr *= tf.minimum(1.0, step / self.warmup_steps)
            else:
                assert name == 'rsqrt_decay'
                lr *= tf.rsqrt(tf.maximum(step, self.warmup_steps))
        tf.contrib.summary.scalar('learning_rate', lr)
        return lr

    def get_config(self):
        return {
            'constant': self.constant,
            'schedule': self.schedule,
            'warmup_steps': self.warmup_steps,
        }


def min_for(dtype):
    return tf.float16.min if dtype == tf.float16 else -1e9


def ones_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = tf.constant(band, tf.float32)
    else:
        band = tf.matrix_band_part(
            tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
            tf.cast(num_upper, tf.int64))
        if out_shape:
            band = tf.reshape(band, out_shape)
    return band


class PadRemover:
    def __init__(self, mask):
        self.ids = None
        self.origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            mask = tf.reshape(mask, [-1])
            self.ids = tf.to_int32(tf.where(mask < 1e-9))
            self.origin = tf.shape(mask)[:1]

    def remove(self, x):
        with tf.name_scope("pad_reduce/remove"):
            return tf.gather_nd(x, indices=self.ids)

    def restore(self, x):
        sh = tf.concat([self.origin, K.int_shape(x)[1:]], axis=0),
        with tf.name_scope("pad_reduce/restore"):
            return tf.scatter_nd(indices=self.ids, updates=x, shape=sh)


def log_confusion_matrix(epoch, logs):
    names = [str(i) for i in range(params.num_classes)]
    labels = [lb.numpy() for _, lb in ds_test]
    preds = np.argmax(model.predict(ds_test), axis=1)
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    img = _to_image(_to_plot(cm, names))
    with writer.as_default():
        tf.summary.image("Confusion Matrix", img, step=epoch)


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
                # run_quess(sess, ps)


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
