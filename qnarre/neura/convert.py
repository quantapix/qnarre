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
import qnarre.neura.utils as utils

from qnarre.neura import tf
from qnarre.neura.mnist import dset_for, model_for, params

TRAIN = 'train'


def main(_):
    ps = utils.Params(params).init_comps()
    ds = dset_for(ps, TRAIN)
    # with T.distribute.MirroredStrategy().scope():
    mdl = model_for(ps, compiled=True)
    mdl.train_on_batch(ds)
    mp = pth.Path.cwd() / ps.dir_model / ps.model
    assert tf.get_checkpoint_state(str(mp))
    mdl.load_weights(str(mp / TRAIN))
    c = tf.Checkpoint(model=mdl, optimizer=ps.optimizer)
    c.restore(str(mp / TRAIN)).expect_partial()  # .assert_consumed()
    for n, s in tf.list_variables(str(mp)):
        print(n)
    mp2 = pth.Path.cwd() / ps.dir_model / 'mnist_2'
    print('saving...')
    c.save(str(mp2 / TRAIN))
    for n, s in tf.list_variables(str(mp2)):
        print(n)
    assert tf.get_checkpoint_state(str(mp2))
    mdl.load_weights(str(mp2 / 'train-1'))


if __name__ == '__main__':
    from absl import app, logging
    logging.set_verbosity(logging.INFO)  # DEBUG
    app.run(main)
