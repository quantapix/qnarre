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
# !pip install -U tf-nightly-2.0-preview

import tensorflow as tf

from datetime import datetime
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import tracking


def trackable(tr1, v):
    c = tf.train.Checkpoint(tr1=tr1)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    c.restore(p).expect_partial()
    if p:
        print(f'restored from: {p}')
        print(f'others are: {m.checkpoints}')
    else:
        print('start from scratch')
    print(f'value before: {v.numpy()}')
    v.assign_add(1)
    m.save()


def autotrackable(tr2, tracked, untracked):
    c = tf.train.Checkpoint(tr2=tr2)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    c.restore(p).expect_partial()
    if p:
        print(f'restored from: {p}')
    print(f'values before: {tracked.numpy()}, {untracked.numpy()}')
    tracked.assign_add(1000)
    m.save()
    print(f'value as saved: {tracked.numpy()}')


def listing():
    c = tf.train.Checkpoint()
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'names and shapes list: {vs}')
    n, _ = vs[-1]
    v = tf.train.load_variable(p, n)
    print(f'loaded value: {v} for name: {n}')
    c = tf.train.load_checkpoint(p)
    ts = c.get_variable_to_dtype_map()
    ss = c.get_variable_to_shape_map()
    print(f'checkpoint types: {ts} and shapes: {ss}')


def deleting(tr2):
    c = tf.train.Checkpoint(tr2=tr2)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    c.restore(m.latest_checkpoint)
    c.tr2.deleted = tf.Variable(-1)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'list deleted: {vs}')
    del c.tr2.deleted
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'deleted IS DELETED: {vs}')


def containers(tr3):
    c = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'containers: {vs}')


def sharing(tr3):
    c = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    c.restore(m.latest_checkpoint).assert_consumed()
    v1 = tr3.br_list[0].v
    v2 = tr3.br_list[1].v
    vd1 = tr3.br_dict['br1'].v
    vd2 = tr3.br_dict['br2'].v
    vd3 = tr3.br_dict['br3'].v
    print(f'all fives: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')
    v1.assign_add(5)
    v2.assign_add(5)
    vd3.assign_add(5)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'shared not repeated: {vs}')
    v1.assign_add(-10)
    v2.assign_add(-10)
    vd3.assign_add(-10)
    print(f'all zeros: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')
    c2 = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c2, '/tmp/q/trackable', max_to_keep=2)
    c2.restore(m.latest_checkpoint).assert_consumed()
    print(f'all tens: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')


class Module(tf.Module):
    sub = None

    def __init__(self, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.v = tf.Variable(1, name='m_v')

    def __str__(self):
        s = f'n: {self.name}, v: {self.v.numpy()}'
        if self.sub:
            s += f', s: ({self.sub})'
        return s

    @tf.Module.with_name_scope
    def __call__(self):
        if self.sub is None:
            y = tf.constant(100)
        else:
            y = self.sub()
        y = tf.math.add(y, self.v)
        self.v.assign(y)
        return y


def modules(mod):
    vs = [v.name for v in mod.variables]
    ms = [m.name for m in mod.submodules]
    print(f'mod variables: {vs}, submodules: {ms}')
    c = tf.train.Checkpoint(module=mod)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    mod()
    print(mod)
    m.save()
    mod()
    print(mod)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'containers: {vs}')
    c.restore(p)
    print(f'restored: {mod}')


def graph(tracer):
    s = datetime.now().strftime('%Y%m%d-%H%M%S')
    d = f'/tmp/q/logs/func/{s}'
    w = tf.summary.create_file_writer(d)
    tf.summary.trace_on(graph=True)  # , profiler=True)
    tracer()
    with w.as_default():
        tf.summary.trace_export(name="trace", step=0, profiler_outdir=d)


class Layer(tf.keras.layers.Layer):
    def __init__(self, sub=None, **kw):
        super().__init__(**kw)
        self.sub = sub

    def __str__(self):
        s = f'n: {self.name}, v: {self.v.numpy()}'
        if self.sub:
            s += f', s: ({self.sub})'
        return s

    def build(self, input_shape):
        self.v = self.add_weight(name='l_v',
                                 shape=[],
                                 dtype=tf.int32,
                                 initializer=tf.ones_initializer)
        return super().build(input_shape)

    def call(self, x):
        if self.sub is None:
            y = x
        else:
            y = self.sub(x)
        y = tf.math.add(y, self.v)
        self.v.assign(tf.reduce_sum(y))
        return y


def models(mod, lay):
    print(mod.summary())
    vs = [v.name for v in mod.variables]
    ts = [t.name for t in mod.trainable_variables]
    ms = [m.name for m in mod.submodules]
    print(f'lay variables: {vs}, trainables: {ts}, submodules: {ms}')
    d = tf.constant([100, 100])
    mod(d)
    print(lay)
    c = tf.train.Checkpoint(model=mod)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    m.save()
    mod(d)
    print(lay)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'containers: {vs}')
    c.restore(p)
    print(f'restored: {lay}')


def main(_):
    tr1 = base.Trackable()
    v = tf.Variable(1)
    tr1._track_trackable(v, name='tr1_v')
    for _ in range(3):
        trackable(tr1, v)

    tr2 = tracking.AutoTrackable()
    tracked, untracked = tf.Variable(1000), tf.Variable(0)
    tr2.v = tracked
    with base.no_automatic_dependency_tracking_scope(tr2):
        tr2.untracked = untracked
    for _ in range(2):
        autotrackable(tr2, tracked, untracked)

    listing()

    deleting(tr2)

    tr3 = tracking.AutoTrackable()
    br1 = tracking.AutoTrackable()
    br1.v = tf.Variable(5)
    br2 = tracking.AutoTrackable()
    br2.v = tf.Variable(5)
    tr3.br_list = [br1, br2]
    br3 = tracking.AutoTrackable()
    br3.v = tf.Variable(5)
    tr3.br_dict = {'br3': br3}
    containers(tr3)

    tr3.br_dict = {'br1': br1, 'br2': br2, 'br3': br3}
    sharing(tr3)

    mod1 = Module('m1')
    mod1.sub = Module('m2')
    mod1.sub.sub = Module('m3')
    modules(mod1)

    # @tf.function
    # def tracer1():
    #     return mod1()

    # graph(tracer1)

    ins = [tf.keras.Input(shape=(), dtype=tf.int32)]
    lay = Layer(name='l1', sub=Layer(name='l2', sub=Layer(name='l3')))
    outs = [lay(ins)]
    mod2 = tf.keras.Model(name='m2', inputs=ins, outputs=outs)
    models(mod2, lay)

    @tf.function
    def tracer2():
        return mod2(tf.constant([100, 100]))

    graph(tracer2)


if __name__ == '__main__':
    from absl import app
    app.run(main)
