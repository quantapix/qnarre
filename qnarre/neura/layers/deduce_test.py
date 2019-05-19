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
# pytest -s qnarre/neura/layers/embed_test.py

import numpy as np

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad

import qnarre.neura.utils as U

from qnarre.neura import tf
from qnarre.neura.layers.embed import TokEmbed

params = dict(
    PAD=0,
    brackets=None,
    dim_embed=4,
    dim_hidden=8,
    emb_one_hot=None,
    num_toks=16,
    tok_types=4,
    len_src=3,
    len_tgt=3,
    pos_max_len=None,
    pos_max=1.0e4,
    pos_min=1.0,
    pos_start=0,
)

ps = U.Params(params).init_comps()


def test_tokembed():
    e = TokEmbed(ps)
    e.build((1, 5))
    src = tf.constant([1, 2, 0, 3, 0], shape=(1, 5))
    e.call(src)
    ps.emb_one_hot = True
    e = TokEmbed(ps)
    e.build((1, 5))
    e.call(src)


def test_w_grad():
    e = TokEmbed(ps)
    e.build((None, 3))
    ins = tf.constant([[0, 1, 0]], dtype='int32')
    with tf.GradientTape() as tape:
        out = e(ins)
    print('===', out, e.weights)
    gs = tape.gradient(out, e.weights)
    opt = adagrad.AdagradOptimizer(0.1)
    opt.apply_gradients(zip(gs, e.weights))
    print('###', len(gs), 1)

"""
class EmbedTest(keras_parameterized.TestCase):
    @keras_parameterized.run_all_keras_modes
    def test_embedding(self):
        if tf_test_util.is_gpu_available():
            self.skipTest('Only test embedding on CPU.')
        testing_utils.layer_test(TokEmbed,
                                 kwargs={'ps': ps},
                                 input_shape=(1, 3),
                                 input_dtype='int32',
                                 expected_output_dtype='float32')

    @keras_parameterized.run_all_keras_modes
    def test_correctness(self):
        lay = TokEmbed(ps)
        mod = tf.Sequential([lay])
        lay.set_weights([np.array([[1, 1], [2, 2]])])
        mod.run_eagerly = testing_utils.should_run_eagerly()
        outputs = mod.predict(np.array([[0, 1, 0]], dtype='int32'))
        self.assertAllClose(outputs, [[[1, 1], [2, 2], [1, 1]]])

    @tf_test_util.run_in_graph_and_eager_modes
    def test_eager_gpu_cpu(self):
        lay = TokEmbed(ps)
        lay.build((None, 2))
        ins = tf.constant([[0, 1, 0]], dtype='int32')
        with tf.GradientTape() as tape:
            out = lay(ins)
        gs = tape.gradient(out, lay.weights)
        opt = adagrad.AdagradOptimizer(0.1)
        opt.apply_gradients(zip(gs, lay.weights))
        self.assertAllEqual(len(gs), 1)


if __name__ == '__main__':
    test.main()
"""
