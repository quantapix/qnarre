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
import pathlib as pth
import datetime as dt

from absl import flags
from tensorboard.plugins import hparams

import qnarre.neura as Q

_params = dict(
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-7,
    adam_lr=0.001,
    initer_stddev=0.02,
    regular_l1=0,
    regular_l2=0,
)


class Params:
    def __init__(self, params, **kw):
        f = 'channels_' + 'first' if Q.is_built_with_cuda() else 'last'
        f = kw.pop('data_format', f)
        self.override(params, data_format=f, **kw)

    @property
    def hparams(self):
        return {'optimizer': self.optimizer}

    def override(self, params, **kw):
        ps = _params.copy()
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

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def init_comps(self):
        ir = Q.TruncatedNormal(stddev=self.initer_stddev)
        rr = None
        if self.regular_l1 or self.regular_l2:
            rr = Q.L1L2(self.regular_l1, self.regular_l2)
        op = Q.Adam(learning_rate=self.adam_beta1,
                    beta_1=self.adam_beta1,
                    beta_2=self.adam_beta2,
                    epsilon=self.adam_epsilon)
        ls = Q.SparseCategoricalCrossentropy(from_logits=True)
        ms = Q.SparseCategoricalAccuracy()
        ffn_act = _activation(self.ffn_act)
        hidden_act = _activation(self.hidden_act)
        self.update(
            initializer=ir,
            regularizer=rr,
            optimizer=op,
            losses=ls,
            metrics=ms,
            ffn_act=ffn_act,
            hidden_act=hidden_act,
            float_min=_float_min(),
        )
        return self


def _float_min():
    f = Q.floatx()
    return Q.float16.min if f == 'float16' else Q.float32.min


def _gelu(x):
    c = Q.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * Q.pow(x, 3))))
    c = (c + 1.0) * 0.5
    return x * c


def _activation(name):
    if isinstance(name, str):
        n = name.lower()
        if n == 'gelu':
            return _gelu
        if n == 'relu':
            return Q.Relu
        if n == 'tanh':
            return Q.Tanh
        assert n == 'linear'
        name = None
    return name


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
        return tuple([Q.as_dtype(self.dtypes[k]) for k in self.keys])

    @property
    def tf_shapes(self):
        return tuple([Q.TensorShape(self.shapes[k]) for k in self.keys])

    def input_kw(self, key):
        return dict(shape=self.shapes[key], dtype=self.dtypes[key])


def train_sess(params, model_fn, dset_fn, cbacks=None, sid=None):
    PS = params
    # with T.distribute.MirroredStrategy().scope():
    m = model_fn(PS)
    ds_train = dset_fn('train', PS)
    ds_test = dset_fn('test', PS)
    sp = pth.Path(PS.save_dir)
    if sp.exists():
        m.train_on_batch(ds_train[:1])
        m.load_weights(sp)
    m.summary()
    sid = sid or dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    p = PS.log_dir + '/train/' + sid
    writer = Q.summary.create_file_writer(p)
    sum_s = hparams.summary.session_start_pb(hparams=PS.hparams)
    cbs = cbacks or []
    cbs.append(
        Q.TensorBoard(log_dir=p,
                      histogram_freq=1,
                      embeddings_freq=0,
                      update_freq='epoch'))
    # cbacKS.append(
    #     KC.EarlyStopping(
    #         monitor='val_loss', min_delta=1e-2, patience=2, verbose=True))
    if sp.exists():
        cbs.append(
            Q.ModelCheckpoint(model_save_path=sp,
                              save_best_only=True,
                              monitor='val_loss',
                              verbose=True))
    hist = m.fit(ds_train,
                 callbacks=cbacks,
                 epochs=PS.train_epochs,
                 validation_data=ds_test)
    print(f'History: {hist.history}')
    if sp.exists():
        m.save_weights(sp, save_format='tf')
    loss, acc = m.evaluate(ds_test)
    print(f'\nTest loss, acc: {loss}, {acc}')

    with writer.as_default():
        e = Q.Event(summary=sum_s).SerializeToString()
        Q.import_event(e)
        Q.scalar('accuracy', acc, step=1, description='Accuracy')
        sum_e = hparams.summary.session_end_pb(hparams.api_pb2.STATUS_SUCCESS)
        e = Q.Event(summary=sum_e).SerializeToString()
        Q.import_event(e)


def train_loop(params, model_fn, dset_fn, cbacks=None):
    PS = params
    nus = [16, 32, 512]
    drs = [0.1, 0.2]
    opts = ['adam', 'sgd']
    writer = Q.create_file_writer(PS.log_dir + '/train')
    with writer.as_default():
        s = None  # _to_summary_pb(nus, drs, opts)
        e = Q.Event(summary=s).SerializeToString()
        Q.import_event(e)
    for nu in nus:
        for dr in drs:
            for opt in opts:
                kw = {'num_units': nu, 'dropout_rate': dr, 'optimizer': opt}
                sid = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
                print(f'--- Running session {sid}:', kw)
                PS.update(**kw)
                train_sess(PS, model_fn, dset_fn, cbacks, sid=sid)
    return


"""
class LRSchedule(KS.optimizers.schedules.LearningRateSchedule):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.constant = PS.lr_constant
        self.schedule = PS.lr_schedule
        self.warmup_steps = PS.lr_warmup_steps

    def __call__(self, step):
        lr = T.constant(1.0)
        for name in [n.strip() for n in self.schedule.split('*')]:
            if name == 'constant':
                lr *= self.constant
            elif name == 'linear_warmup':
                lr *= T.minimum(1.0, step / self.warmup_steps)
            else:
                assert name == 'rsqrt_decay'
                lr *= T.rsqrt(T.maximum(step, self.warmup_steps))
        T.contrib.summary.scalar('learning_rate', lr)
        return lr

    def get_config(self):
        return {
            'constant': self.constant,
            'schedule': self.schedule,
            'warmup_steps': self.warmup_steps,
        }


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


class PadRemover:
    def __init__(self, mask):
        self.ids = None
        self.origin = None
        with T.name_scope("pad_reduce/get_ids"):
            mask = T.reshape(mask, [-1])
            self.ids = Q.cast(T.where(mask < 1e-9), 'int32')
            self.origin = T.shape(mask)[:1]

    def remove(self, x):
        with T.name_scope("pad_reduce/remove"):
            return T.gather_nd(x, indices=self.ids)

    def restore(self, x):
        sh = T.concat([self.origin, Q.int_shape(x)[1:]], axis=0),
        with T.name_scope("pad_reduce/restore"):
            return T.scatter_nd(indices=self.ids, updates=x, shape=sh)


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
    cm = N.around(cm.astype('float') / cm.sum(axis=1)[:, N.newaxis],
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
