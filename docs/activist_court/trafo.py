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
import samples as qs
import layers as ql
import utils as qu

ks = tf.keras


def model_for(ps, group):
    x = []
    for _ in (qd.ENC, qd.DEC, qd.TGT):
        x.append(ks.Input(shape=(), dtype='int32'))
        x.append(ks.Input(shape=(), dtype='int64'))
    for _ in (qd.EMT, qd.DMT):
        x.append(ks.Input(shape=(), dtype='int32'))
    y = ql.ToRagged()(x)
    yt, ym = ql.Tokens(ps)(y), ql.Metas(ps)(y)
    xe, xd = yt[:2] + ym[:1], yt[2:] + ym[1:]
    embed = ql.Embed(ps)
    ye = ql.Encode(ps)(embed(xe))[0]
    decode = ql.Decode(ps)
    if group in (qs.YNS, qs.YNX):
        y = decode(embed(xd) + [ye])
        y = ql.Debed(ps)(y)
    elif group in (qs.MSK, qs.MSX):
        y = ql.Deduce(ps, embed, decode)(xd + [ye])
    if group in (qs.QAS, qs.FIX):
        y = decode(embed(xd) + [ye])
        y = ql.Output(ps, group)(y)
    m = ks.Model(inputs=x, outputs=[y])
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
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
    num_epochs=2,
    num_heads=3,
    num_rounds=2,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    print_toks=False,
    width_dec=40,
    width_enc=50,
)

params.update(
    loss=qu.Loss(),
    metric=qu.Metric(),
)


def main(ps, fn, groups=None, count=None):
    groups = groups or qs.groups
    for r in range(ps.num_rounds):
        for g in groups:
            print(f'\nRound {r + 1}, group {g}...\n=======================')
            fn(ps, qd.dset_for(ps, g, count=count), model_for(ps, g))


if __name__ == '__main__':
    ps = qu.Params(**params)
    ps.is_training = True
    # qu.train_eager(ps, qd.dset_for(ps, 'msk', count=10), model_for(ps))
    main(ps, qu.train_eager, groups=(qs.YNS, qs.MSK, qs.QAS), count=10)
