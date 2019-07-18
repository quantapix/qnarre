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

import datasets as qd
import layers as ql
import utils as qu

ks = tf.keras


def model_for(ps):
    x = []
    for _ in ('encode', 'decode', 'target'):
        x.append(ks.Input(shape=(), dtype='int32'))
        x.append(ks.Input(shape=(), dtype='int64'))
    for _ in ('e_meta', 'd_meta'):
        x.append(ks.Input(shape=(), dtype='int32'))
    y = ql.ToRagged()(x)
    yt, ym = ql.Tokens(ps)(y), ql.Metas(ps)(y)
    xe, xd = yt[:2] + ym[:1], yt[2:] + ym[1:]
    embed = ql.Embed(ps)
    ye = ql.Encode(ps)(embed(xe))[0]
    decode = ql.Decode(ps)
    yb = ql.Debed(ps)(decode(embed(xd) + [ye]))
    yc = ql.Deduce(ps, embed, decode)(xd + [ye])
    m = ks.Model(inputs=x, outputs=[yb, yc])
    m.compile(optimizer=ps.optimizer,
              loss={
                  'debed': ps.loss,
                  'deduce': ps.loss
              },
              metrics={
                  'debed': [ps.metric],
                  'deduce': [ps.metric]
              })
    print(m.summary())
    return m


params = dict(
    activ_concl='gelu',
    dim_attn=4,
    dim_attn_qk=None,
    dim_attn_v=None,
    dim_batch=5,
    dim_concl=150,
    dim_hidden=6,
    dim_hist=5,
    dim_metas=len(qd.metas),
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    drop_attn=None,
    drop_concl=None,
    drop_hidden=0.1,
    initer_stddev=0.02,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=5,
    num_heads=3,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    print_toks=False,
    width_dec=15,
    width_enc=25,
)

params.update(
    loss=qu.Loss(),
    metric=qu.Metric(),
)

# groups = ('yns', 'ynx', 'msk', 'msx', 'cls', 'clx', 'qas', 'rev', 'gen', 'fix')
groups = ('yns', 'ynx', 'msk', 'msx', 'cls', 'clx', 'rev', 'fix')


if __name__ == '__main__':
    ps = qu.Params(**params)
    ps.is_training = True
    # qu.train_graph(ps, qd.dset_for(ps), model_for(ps))
    # qu.train_eager(ps, qd.dset_for(ps, count=10), model_for(ps))
    qu.train_eager(ps, qd.dset_for(ps, 'msk'), model_for(ps))
