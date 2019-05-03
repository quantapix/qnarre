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
# https://arxiv.org/pdf/1701.06548.pdf
# https://arxiv.org/pdf/1607.06450.pdf
# https://arxiv.org/pdf/1606.08415.pdf

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.feeds.dset.trafo import dset as dset


def dset_for(PS, kind):
    ds = dset(PS, kind)
    n = 1000
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(PS.batch_size)
    return ds


def model_for(PS, full=False):
    ctx = Q.Input(shape=(PS.ctx_len, ), dtype='int32')
    typ = Q.Input(shape=(PS.ctx_len, ), dtype='int32')
    tgt = Q.Input(shape=(PS.tgt_len, ), dtype='int32')
    ins = [ctx, typ, tgt]
    y = L.Trafo(PS)(ins)
    m = Q.Model(name='TrafoModel', inputs=ins, outputs=[y])
    if full:
        m.compile(optimizer=PS.optimizer, loss=PS.losses, metrics=[PS.metrics])
    print(m.summary())
    return m


params = dict(
    attn_drop=0.1,
    attn_heads=2,
    attn_k_size=4,
    attn_type=None,
    attn_v_size=4,
    batch_size=4,
    causal_refl=False,
    ctx_len=16,
    dec_layers=None,
    enc_layers=None,
    ffn_act='gelu',
    ffn_drop=None,
    ffn_size=256,
    ffn_type=None,
    hidden_act='gelu',
    hidden_drop=0.1,
    hidden_size=8,
    max_pos=None,
    pos_embed='timing',
    prox_bias=True,
    refl_type=None,
    stack_layers=2,
    tgt_len=16,
    token_types=8,
    vocab_size=20,
)

params.update(
    data_dir='.data/trafo',
    log_dir='.model/trafo/logs',
    model_dir='.model/trafo',
    save_dir='.model/trafo/save',
)


def main(_):
    PS = U.Params(params).init_comps()
    model = model_for(PS)

    @Q.function
    def train_step(x, y):
        with Q.GradientTape() as tape:
            logits = model(x)
            loss = PS.losses(y, logits)
            acc = PS.metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        PS.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    def train():
        step, loss, acc = 0, 0.0, 0.0
        for x, y in dset_for(PS, 'train'):
            step += 1
            loss, acc = train_step(x, y)
            if Q.equal(step % 10, 0):
                m = PS.metrics.result()
                Q.print('Step:', step, ', loss:', loss, ', acc:', m)
        return step, loss, acc

    step, loss, acc = train()
    print('Final step:', step, ', loss:', loss, ', acc:', PS.metrics.result())


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('src_len', None, '')
    from absl import app
    app.run(main)
