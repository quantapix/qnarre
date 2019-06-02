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

import pathlib as pth
import datetime as dt

from qnarre.neura import tf
from tensorboard.plugins import hparams
from tensorboard.plugins.hparams import summary as tb_summary


def session_for(ps, sid=None):
    if ps.predict_run:
        sess = eager_pred if ps.eager_mode else predict
    else:
        if ps.eval_only:
            sess = eager_eval if ps.eager_mode else evaluate
        else:
            sess = eager_train if ps.eager_mode else train
    sid = sid or datetime.now().strftime('%Y%m%d-%H%M%S')
    return lambda *args, **kw: sess(sid, ps, *args, **kw)


TRAIN = 'train'


def eager_train(sid, ps, dset_fn, model_fn, cbacks=None):
    dset = dset_fn(ps, TRAIN)
    # dset_test = dset_fn(ps, 'test')
    model = model_fn(ps)

    def step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = ps.losses(y, logits)
            loss += sum(model.losses)
            acc = ps.metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        ps.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    @tf.function
    def epoch():
        s, loss, acc = 0, 0.0, 0.0
        for x, y in dset:
            s += 1
            loss, acc = step(x, y)
            if tf.equal(s % 10, 0):
                m = ps.metrics.result()
                tf.print('Step:', s, ', loss:', loss, ', acc:', m)
        return loss, acc

    for e in range(ps.train_epochs):
        loss, acc = epoch()
        print(f'Epoch {e} loss:', loss, ', acc:', acc)


def train(sid, ps, dset_fn, model_fn, cbacks=None):
    ds = dset_fn(ps, TRAIN)
    # with T.distribute.MirroredStrategy().scope():
    mdl = model_fn(ps, compiled=True)
    mp = pth.Path.cwd() / ps.dir_model / ps.model
    if mp.exists() and tf.get_checkpoint_state(str(mp)):
        mdl.train_on_batch(ds)
        mdl.load_weights(str(mp / TRAIN))
    lp = pth.Path.cwd() / ps.dir_log / ps.model
    if lp.exists():
        sumy = tf.create_file_writer(str(lp / TRAIN / sid))
        sum_s = tb_summary.session_start_pb(hparams=ps.hparams)
    cbs = cbacks or []
    if lp.exists():
        cbs.append(
            tf.TensorBoard(log_dir=str(lp / TRAIN / sid),
                           histogram_freq=1,
                           embeddings_freq=0,
                           update_freq='epoch'))
    cbs.append(
        tf.EarlyStopping(monitor='val_loss',
                         min_delta=1e-2,
                         patience=2,
                         verbose=True))
    if mp.exists():
        cbs.append(
            tf.ModelCheckpoint(str(mp / TRAIN),
                               save_weights_only=True,
                               # save_best_only=True,
                               verbose=True))
    ds_test = dset_fn(ps, 'test')
    hist = mdl.fit(ds,
                   callbacks=cbs,
                   epochs=ps.train_epochs,
                   validation_data=ds_test)
    print(f'History: {hist.history}')
    sp = pth.Path.cwd() / ps.dir_save / ps.model
    if sp.exists():
        tf.export_saved_model(mdl, str(sp))
    loss, acc = mdl.evaluate(ds_test)
    print(f'\nEval loss, acc: {loss}, {acc}')
    """
    with sumy.as_default():
        e = tf.Event(summary=sum_s).SerializeToString()
        tf.import_event(e)
        tf.scalar('accuracy', acc, step=1, description='Accuracy')
        sum_e = tb_summary.session_end_pb(hparams.api_pb2.STATUS_SUCCESS)
        e = tf.Event(summary=sum_e).SerializeToString()
        tf.import_event(e)
    """


def evaluate(sid, ps, dset_fn, model_fn, cbacks=None):
    ds = dset_fn(ps, 'test')
    # with T.distribute.MirroredStrategy().scope():
    p = str(pth.Path.cwd() / ps.dir_save / ps.model)
    assert tf.contains_saved_model(p)
    mdl = tf.load_from_saved_model(p)
    loss, acc = mdl.evaluate(ds)
    print(f'\nEvaluate loss, acc: {loss}, {acc}')


def predict(sid, ps, dset_fn, model_fn, cbacks=None):
    ds = dset_fn(ps, 'try')
    # with T.distribute.MirroredStrategy().scope():
    p = str(pth.Path.cwd() / ps.dir_save / ps.model)
    assert tf.contains_saved_model(p)
    m = tf.load_from_saved_model(p)


def train_loop(params, model_fn, dset_fn, cbacks=None):
    ps = params
    nus = [16, 32, 512]
    drs = [0.1, 0.2]
    opts = ['adam', 'sgd']
    writer = tf.create_file_writer(ps.dir_log + '/train')
    with writer.as_default():
        s = None  # _to_summary_pb(nus, drs, opts)
        e = tf.Event(summary=s).SerializeToString()
        tf.import_event(e)
    for nu in nus:
        for dr in drs:
            for opt in opts:
                kw = {'num_units': nu, 'dropout_rate': dr, 'optimizer': opt}
                sid = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
                print(f'--- Running session {sid}:', kw)
                ps.update(**kw)
                train_sess(ps, model_fn, dset_fn, cbacks, sid=sid)
    return


"""
names = [str(i) for i in range(ps.num_classes)]
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
