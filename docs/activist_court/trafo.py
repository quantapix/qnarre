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

import tensorflow as tf

import data as qd
import layers as ql
import utils as qu

ks = tf.keras


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = ql.ToRagged()(x)
    y = ql.Frames(ps)(y)
    embed = ql.Embed(ps)
    ye = ql.Encode(ps)(embed(y[:2]))
    yd = ql.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = ql.Probe(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


params = dict(
    dim_attn=None,
    dim_attn_k=None,
    dim_attn_v=None,
    dim_batch=5,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    drop_attn=None,
    drop_hidden=0.1,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=5,
    num_heads=None,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    print_frames=False,
    width_dec=15,
    width_enc=25,
)

params.update(
    loss=qu.Loss(),
    metric=qu.Metric(),
)

if __name__ == '__main__':
    ps = qd.Params(**params)
    qu.train_graph(ps, qd.dset_for(ps), model_for(ps))
    # qu.train_eager(ps, qd.dset_for(ps), model_for(ps))
