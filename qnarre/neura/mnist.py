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

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.feeds.dset.mnist import dset as dset


def dset_for(PS, kind):
    ds_1 = dset(PS, kind)
    ds_2 = dset(PS, kind)
    ds_3 = dset(PS, kind)
    n = 50000
    if kind == 'train':
        ds_1 = ds_1.shuffle(n)
        ds_2 = ds_2.shuffle(n)
        ds_3 = ds_3.shuffle(n)
    ds = Q.Dataset.zip((ds_1, ds_2, ds_3))
    ds = ds.map(lambda s, s2, s3: ((s[0], s2[0], s2[1], s3[0], s3[1]), s[1]))
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(PS, full=False):
    ins = [
        Q.Input(shape=(28, 28), dtype='float32'),
        Q.Input(shape=(28, 28), dtype='float32'),
        Q.Input(shape=(1, ), dtype='int32'),
        Q.Input(shape=(28, 28), dtype='float32'),
        Q.Input(shape=(1, ), dtype='int32'),
    ]
    outs = [L.Mnist(PS)(ins)]
    # outs = [Mnist_1(PS)(ins), Mnist_2(PS)(ins), Mnist_3(PS)(ins)]
    m = Q.Model(inputs=ins, outputs=outs)
    if full:
        m.compile(
            optimizer=PS.optimizer,
            loss=PS.losses,
            metrics=[PS.metrics],
            target_tensors=[ins[4]],
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
    print(m.summary())
    return m


params = dict(
    batch_size=64,
    epochs_between_evals=1,
    ffn_act=None,
    hidden_act='relu',
    hidden_drop=0.2,
    hidden_size=512,
    model_name='mlp',
    num_classes=10,
    optimizer='sgd',
    seq_len=28 * 28,
    train_epochs=2,
)

params.update(
    data_dir='.data/mnist',
    log_dir='.model/mnist/logs',
    model_dir='.model/mnist',
    save_dir='.model/mnist/save',
)


def main(_):
    PS = U.Params(params).init_comps()
    dset = dset_for(PS, 'train')
    model = model_for(PS)

    def train_step(x, y):
        with Q.GradientTape() as tape:
            logits = model(x)
            loss = PS.losses(y, logits)
            acc = PS.metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        PS.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    @Q.function
    def train():
        step, loss, acc = 0, 0.0, 0.0
        for x, y in dset:
            step += 1
            loss, acc = train_step(x, y)
            if Q.equal(step % 10, 0):
                m = PS.metrics.result()
                Q.print('Step:', step, ', loss:', loss, ', acc:', m)
        return step, loss, acc

    step, loss, acc = train()
    print('Final step:', step, ', loss:', loss, ', acc:', PS.metrics.result())


def main_2(_):
    PS = U.Params(params).init_comps()
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
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
    from absl import flags
    flags.DEFINE_integer('num_classes', None, '')
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
