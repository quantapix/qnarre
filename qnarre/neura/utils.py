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

import numpy as np

from absl import flags
# from tensorboard.plugins import hparams

from qnarre.neura import tf


def big_neg():
    f = tf.floatx()
    return tf.float16.min if f == 'float16' else -1e9


def gelu(x):
    c = tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    c = (c + 1.0) * 0.5
    return x * c


def pos_timing(dim, end):
    t = tf.range(end - 1, -1, -1, dtype=tf.floatx())
    f = tf.range(0, dim, 2.0, dtype=tf.floatx())
    f = 1 / (10000**(f / dim))
    t = tf.einsum('i,d->id', t, f)
    return tf.concat([tf.sin(t), tf.cos(t)], axis=-1)


def pos_timing_2(dim, end, p_max, p_min, p_start):
    t = tf.range(end, dtype=tf.floatx()) + p_start
    assert dim % 2 == 0
    n = dim // 2
    f = np.log(p_max / p_min) / max(n - 1, 1)
    f = tf.range(n, dtype=tf.floatx()) * -f
    f = tf.exp(f) * p_min
    t = tf.einsum('i,d->id', t, f)
    return tf.concat([tf.sin(t), tf.cos(t)], axis=-1)


def load_flags():
    # flags.DEFINE_float('stop_threshold', None, '')
    # flags.DEFINE_integer('checkpoint_steps', None, '')
    # flags.DEFINE_integer('epochs_between_evals', None, '')
    # flags.DEFINE_integer('eval_batch_size', None, '')
    # flags.DEFINE_integer('eval_steps', None, '')
    # flags.DEFINE_integer('iters_per_loop', None, '')
    # flags.DEFINE_integer('warmup_steps', None, '')
    # flags.DEFINE_string('log_dir', None, '')
    df = ['channels_first', 'channels_last']
    flags.DEFINE_bool('eager_mode', False, '')
    flags.DEFINE_bool('eval_only', False, '')
    flags.DEFINE_bool('predict_run', False, '')
    flags.DEFINE_enum('data_format', None, df, '')
    flags.DEFINE_integer('batch_size', None, '')
    flags.DEFINE_integer('train_epochs', None, '')
    flags.DEFINE_integer('train_steps', None, '')
    flags.DEFINE_string('dir_data', None, '')
    flags.DEFINE_string('dir_model', None, '')
    flags.DEFINE_string('dir_save', None, '')
    flags.DEFINE_string('model', None, '')


root = dict(
    act_ffnet=None,
    act_hidden=None,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-7,
    adam_lr=0.001,
    dir_data='.data',
    dir_log='.log',
    dir_model='.model',
    dir_save='.save',
    dset=None,
    eager_mode=False,
    epochs_between_evals=1,
    eval_batch_size=None,
    eval_only=False,
    init_stddev=0.02,
    loss_from_logits=True,
    lr_constant=None,
    lr_schedule=None,
    lr_warmup=None,
    model=None,
    optimizer=None,
    predict_run=False,
    regular_l1=0,
    regular_l2=0,
    sgd_lr=0.001,
    sgd_momentum=0.0,
    sgd_nesterov=False,
    tokenizer=None,
    train_epochs=2,
    train_steps=None,
    vocab_path='vocab.tfrecords',
)


class Params:
    def __init__(self, params, **kw):
        f = 'channels_' + 'first' if tf.is_built_with_cuda() else 'last'
        f = kw.pop('data_format', f)
        self.override(params, data_format=f, **kw)

    @property
    def hparams(self):
        return {}  # 'optimizer': self.optimizer}

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def override(self, params, **kw):
        ps = root.copy()
        ps.update(**params)
        ps.update(**kw)
        f = flags.FLAGS
        for k, v in ps.items():
            if hasattr(f, k):
                v = getattr(f, k)
                if v:
                    ps[k] = v
        self.update(**ps)
        return self

    def cfg_items(self, *keys):
        for k in keys:
            yield k, getattr(self, k)

    def init_comps(self):
        rr = None
        if self.regular_l1 or self.regular_l2:
            rr = tf.L1L2(self.regular_l1, self.regular_l2)
        self.update(
            initializer=tf.TruncatedNormal(stddev=self.init_stddev),
            regularizer=rr,
            optimizer=self._optimizer(self.optimizer),
            # losses=tf.MeanSquaredError(),
            losses=tf.SparseCategoricalCrossentropy(
                from_logits=self.loss_from_logits),
            metrics=tf.SparseCategoricalAccuracy(),
            act_ffnet=self._activation(self.act_ffnet),
            act_hidden=self._activation(self.act_hidden),
        )
        return self

    @staticmethod
    def _activation(act):
        if isinstance(act, str):
            n = act.lower()
            if n == 'gelu':
                return gelu
            if n == 'relu':
                return tf.Relu
            if n == 'tanh':
                return tf.Tanh
            assert n == 'linear'
            return None
        return act

    def _optimizer(self, opt):
        opt = 'sgd' if opt is None else opt
        if isinstance(opt, str):
            n = opt.lower()
            if n == 'adam':
                return tf.Adam(learning_rate=self.adam_lr,
                               beta_1=self.adam_beta1,
                               beta_2=self.adam_beta2,
                               epsilon=self.adam_epsilon)
            assert n == 'sgd'
            return tf.SGD(learning_rate=self.sgd_lr,
                          momentum=self.sgd_momentum,
                          nesterov=self.sgd_nesterov)
        return opt


class LearningRateSchedule(tf.LearningRateSchedule):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.constant = PS.lr_constant
        self.schedule = PS.lr_schedule
        self.warmup = PS.lr_warmup

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
        tf.scalar('learning_rate', lr)
        return lr

    def get_config(self):
        return {
            'constant': self.constant,
            'schedule': self.schedule,
            'warmup': self.warmup,
        }


class LRFinder:
    # https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

    def __init__(self, optimizer, max_step, init_value=1e-8, final_value=1.):
        self.gamma = (final_value / init_value)**(1 / max_step)
        self.optimizer = optimizer
        self.init_value = init_value
        optimizer.param_groups[0]['lr'] = init_value

    def step(self, step):
        lr = self.init_value * self.gamma**step
        self.optimizer.param_groups[0]['lr'] = lr


class Features:
    def __init__(self, specs, **kw):
        self.keys = []
        self.dtypes = {}
        self.shapes = {}
        for k, v in specs.items():
            assert not hasattr(self, k)
            setattr(self, k, v['name'])
            k = v['name']
            self.keys.append(k)
            self.dtypes[k] = v['dtype']
            self.shapes[k] = v['shape']

    @property
    def tf_dtypes(self):
        return tuple([tf.as_dtype(self.dtypes[k]) for k in self.keys])

    @property
    def tf_shapes(self):
        return tuple([tf.TensorShape(self.shapes[k]) for k in self.keys])

    def input_kw(self, key):
        return dict(shape=self.shapes[key], dtype=self.dtypes[key])


"""
def ones_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = N.tri(cols, rows, num_lower).T
        upper_mask = N.tri(rows, cols, num_upper)
        band = N.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = T.constant(band, T.float32)
    else:
        band = T.matrix_band_part(T.ones([rows, cols]),
                                  T.cast(num_lower, T.int64),
                                  T.cast(num_upper, T.int64))
        if out_shape:
            band = T.reshape(band, out_shape)
    return band


def log_confusion_matrix(epoch, logs):
    names = [str(i) for i in range(params.num_classes)]
    labels = [lb.numpy() for _, lb in ds_test]
    preds = N.argmax(model.predict(ds_test), axis=1)
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    img = _to_image(_to_plot(cm, names))
    with writer.as_default():
        T.summary.image("Confusion Matrix", img, step=epoch)


def _to_plot(cm, names):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = N.arange(len(names))
    plt.xticks(ticks, names, rotation=45)
    plt.yticks(ticks, names)
    cm = N.around(cm.astype('float') / cm.sum(axis=1)[:, N.psaxis],
                  decimals=2)
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
    img = T.image.decode_png(buf.getvalue(), channels=4)
    img = T.expand_dims(img, 0)
    return img


def _to_summary_pb(num_units_list, dropout_rate_list, optimizer_list):
    nus_val = struct_pb2.ListValue()
    nus_val.extend(num_units_list)
    drs_val = struct_pb2.ListValue()
    drs_val.extend(dropout_rate_list)
    opts_val = struct_pb2.ListValue()
    opts_val.extend(optimizer_list)
    return hparams.summary.experiment_pb(
        hparam_infos=[
            hparams.api_pb2.HParamInfo(name='num_units',
                                       display_name='Number of units',
                                       type=hparams.api_pb2.DATA_TYPE_FLOAT64,
                                       domain_discrete=nus_val),
            hparams.api_pb2.HParamInfo(name='dropout_rate',
                                       display_name='Dropout rate',
                                       type=hparams.api_pb2.DATA_TYPE_FLOAT64,
                                       domain_discrete=drs_val),
            hparams.api_pb2.HParamInfo(name='optimizer',
                                       display_name='Optimizer',
                                       type=hparams.api_pb2.DATA_TYPE_STRING,
                                       domain_discrete=opts_val)
        ],
        metric_infos=[
            hparams.api_pb2.MetricInfo(
                name=hparams.api_pb2.MetricName(tag='accuracy'),
                display_name='Accuracy'),
        ])


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    import re
    import collections as co
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = co.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = T.train.list_variables(init_checkpoint)

    assignment_map = co.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)
"""
