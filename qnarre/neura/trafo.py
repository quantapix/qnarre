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

import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.neura import tf
from qnarre.feeds.dset.trafo import dset as dset
from qnarre.neura.session import session_for


def dset_for(ps, kind):
    ds = dset(ps, kind)
    n = 1000
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(ps.batch_size)
    return ds


def model_for(ps, compiled=False):
    ctx = tf.Input(shape=(ps.ctx_len, ), dtype='int32')
    typ = tf.Input(shape=(ps.ctx_len, ), dtype='int32')
    tgt = tf.Input(shape=(ps.tgt_len, ), dtype='int32')
    ins = [ctx, typ, tgt]
    outs = [L.Trafo(ps)(ins)]
    m = tf.Model(name='TrafoModel', inputs=ins, outputs=outs)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
        )
    print(m.summary())
    return m


_params = dict(
    act_hidden='gelu',
    attn_heads=2,
    batch_size=4,
    beam_size=None,
    brackets=None,
    causal_refl=False,
    ctx_len=16,
    dec_layers=None,
    dim_attn=8,
    dim_embed=None,
    dim_hidden=16,
    dim_attn_k=None,
    dim_attn_v=None,
    drop_attn=None,
    drop_hidden=0.1,
    emb_one_hot=None,
    enc_layers=None,
    ffn_act='gelu',
    ffn_drop=None,
    ffn_size=256,
    ffn_type=None,
    max_pos=None,
    norm_epsilon=1e-6,
    norm_type='layer',
    num_heads=4,
    num_toks=None,
    pos_embed='timing',
    pos_max=1.0e4,
    pos_min=1.0,
    pos_start=0,
    post_cmd='dan',
    pre_cmd='n',
    prepost_bdims='',
    prepost_drop=None,
    prox_bias=True,
    stack_layers=2,
    tgt_len=None,
    tok_types=8,
)

_params.update(
    data_dir='.data/trafo',
    log_dir='.model/trafo/logs',
    model_dir='.model/trafo',
    save_dir='.model/trafo/save',
)


def main(_):
    ps = U.Params(_params).init_comps()
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('ctx_len', None, '')
    from absl import app
    app.run(main)
