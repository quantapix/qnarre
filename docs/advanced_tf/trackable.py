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
# !pip install tensorflow==2.0.0-beta0

import tensorflow as tf

# from tensorflow.python.module import module
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import tracking


def trackable(tr1, v):
    c = tf.train.Checkpoint(tr1=tr1)
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
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
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
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
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
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
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
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
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'list containers: {vs}')


def sharing(tr3):
    c = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
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
    m = tf.train.CheckpointManager(c2, '/tmp/trackable', max_to_keep=2)
    c2.restore(m.latest_checkpoint).assert_consumed()
    print(f'all tens: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')


class Module(tf.Module):
    sub = None

    def __init__(self, name=None):
        super().__init__(name=name)
        self.v = tf.Variable(0, name='m_v')

    @tf.function
    def __call__(self):
        y = tf.math.add(self.v, tf.constant(1))
        if self.sub is not None:
            y = tf.math.add(y, self.sub())
        self.v.assign(y)
        return y

    def print_all(self):
        print(f'{self.name}: {self.v.numpy()}')
        if self.sub:
            self.sub.print_all()


def modules(mod1):
    # vs = [v.name for v in mod1.variables]
    # ms = [m.name for m in mod1.submodules]
    # print(f'mod variables: {vs}, submodules: {ms}')
    c = tf.train.Checkpoint(mod1=mod1)
    m = tf.train.CheckpointManager(c, '/tmp/trackable', max_to_keep=2)
    mod1()
    mod1.print_all()
    m.save()
    mod1()
    mod1.print_all()
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'list containers: {vs}')
    c.restore(p)
    print('after restore:')
    mod1.print_all()


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
    mod1.sub.sub.sub = Module('m4')
    mod1.sub.sub.sub.sub = Module('m5')
    modules(mod1)
    """
    opt = tf.keras.optimizers.Adam(0.1)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
    to_restore = tf.Variable(tf.zeros([5]))
    print(to_restore.numpy())  # All zeros
    fake_layer = tf.train.Checkpoint(bias=to_restore)
    fake_net = tf.train.Checkpoint(l1=fake_layer)
    new_root = tf.train.Checkpoint(net=fake_net)
    status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
    print(to_restore.numpy())
    """


if __name__ == '__main__':
    from absl import app, flags
    # flags.DEFINE_integer('num_classes', None, '')
    app.run(main)
    # python -m qnarre.neura.mnist -eager_mode
