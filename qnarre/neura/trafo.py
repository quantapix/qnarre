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

from qnarre.neura import tf
from qnarre.neura.utils import Params
from qnarre.neura.layers import Trafo
from qnarre.neura.session import session_for
from qnarre.feeds.dset.trafo import dset as dset


def dset_for(ps, kind):
    ds = dset(ps, kind)
    n = 1000
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(ps.batch_size)
    return ds


def model_for(ps, compiled=False):
    src = tf.Input(shape=(ps.len_src, ), dtype='int32')
    typ = tf.Input(shape=(ps.len_src, ), dtype='int32')
    tgt = tf.Input(shape=(ps.len_tgt, ), dtype='int32')
    ins = [src, typ, tgt, None, None]
    outs = [Trafo(ps)(ins)]
    m = tf.Model(name='TrafoModel', inputs=ins, outputs=outs)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
        )
    print(m.summary())
    return m


params = dict(
    act_ffnet='gelu',
    act_hidden='gelu',
    batch_size=4,
    bdims_prepost='',
    beam_size=None,
    bias_prox=True,
    brackets=None,
    causal_refl=False,
    cmd_post='dan',
    cmd_pre='n',
    dim_attn=8,
    dim_attn_k=None,
    dim_attn_v=None,
    dim_embed=None,
    dim_ffnet=256,
    dim_hidden=16,
    drop_attn=None,
    drop_ffnet=None,
    drop_hidden=0.1,
    drop_prepost=None,
    emb_one_hot=None,
    num_dec_lays=None,
    num_enc_lays=None,
    num_stack_lays=2,
    len_ctx=None,
    len_src=16,
    len_tgt=None,
    max_pos=None,
    norm_epsilon=1e-6,
    norm_type='layer',
    num_heads=4,
    num_toks=None,
    pos_emb='timing',
    pos_max=1.0e4,
    pos_min=1.0,
    pos_start=0,
    tok_typs=8,
)

params.update(
    data_dir='.data/trafo',
    log_dir='.model/trafo/logs',
    model_dir='.model/trafo',
    save_dir='.model/trafo/save',
)


def main(_):
    ps = Params(params).init_comps()
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('len_ctx', None, '')
    from absl import app
    app.run(main)
