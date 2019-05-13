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

import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.neura import tf

params = dict(
    num_toks=None,
    brackets=None,
    dim_hidden=16,
    dim_embed=None,
    initializer='uniform',
    emb_one_hot=True,

    dim_attn=8,
    dim_k=None,
    dim_v=None,
    drop_attn=None,
    drop_hidden=0.1,
    num_heads=4,
)

PS = U.Params(params).init_comps()


def test_owner_none():
    a = L.Attn(PS)
    a.build([(4, 10, 16)])
    src = tf.constant([0.] * (4 * 10 * 16), shape=(4, 10, 16))
    a.call([src])
    bias = tf.constant([0.] * (4 * 10), shape=(4, 10))
    bias = tf.expand_dims(tf.expand_dims(bias, axis=1), axis=3)
    a.call([src, bias])
    ctx = tf.constant([0.] * (4 * 15 * 16), shape=(4, 15, 16))
    a.call([src, bias, None, ctx])


class Owner:
    def __init__(self):
        self.pre = None
        self.post = None
        i = tf.constant([0.] * (4 * 10), shape=(4, 10))
        i = tf.expand_dims(tf.expand_dims(i, axis=1), axis=3)
        self.src_bias = tf.Variable(initial_value=i)
        i = tf.constant([0.] * (4 * 10), shape=(4, 10))
        i = tf.expand_dims(tf.expand_dims(i, axis=1), axis=3)
        self.mem_bias = tf.Variable(initial_value=i)


def test_with_owner():
    a = L.Attn(PS, owner=Owner())
    a.build([(4, 10, 16), (), (4, 18, 16), ()])
    src = tf.constant([0.] * (4 * 10 * 16), shape=(4, 10, 16))
    bias = tf.constant([0.] * (4 * 10), shape=(4, 10))
    bias = tf.expand_dims(tf.expand_dims(bias, axis=1), axis=3)
    mem = tf.constant([0.] * (4 * 15 * 16), shape=(4, 15, 16))
    ctx = tf.constant([0.] * (4 * 15 * 16), shape=(4, 15, 16))
    a.call([src, bias, mem, ctx])


def test_shift():
    a = L.Attn(PS, owner=Owner())
    x = tf.constant([1, 2, 3, 4, 5, 6], shape=(1, 1, 2, 3))
    tf.print(x)
    x = a.shift(x)
    tf.print(x)
