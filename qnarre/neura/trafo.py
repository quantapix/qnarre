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

from absl import flags

from qnarre.neura import utils
from qnarre.neura.session import session_for

from qnarre.feeds.dset.squad import dset as trafo_dset
from qnarre.neura.layers.squad import model as trafo_model
from qnarre.neura.layers.squad import adapter as trafo_adapter


def dset_for(ps, kind):
    ds, feats = trafo_dset(ps, kind)
    if kind == 'train':
        ds = ds.shuffle(10000)
    ds = ds.batch(1 if ps.eager_mode else ps.batch_size)
    ds = ds.map(lambda d: trafo_adapter(ps, feats, d))
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(ps, compiled=False):
    m = trafo_model(ps)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
            # target_tensors=[ins[4]],
        )
    print(m.summary())
    return m


params = dict(
    act_ffnet='gelu',
    act_hidden='gelu',
    batch_size=4,
    bdims_prepost='',
    beam_size=None,
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
    len_ctx=None,
    len_mem=None,
    len_src=16,
    len_tgt=None,
    max_pos=None,
    norm_epsilon=1e-6,
    norm_type='layer',
    num_dec_lays=None,
    num_enc_lays=None,
    num_heads=4,
    num_stack_lays=2,
    num_toks=None,
    pos_max=1.0e4,
    pos_max_len=None,
    pos_min=1.0,
    pos_start=0,
    pos_type='timing',
    proxim_bias=True,
    share_adapt=True,
    share_table=True,
    tok_types=8,
)


def load_params():
    return utils.Params(params).init_comps()


def load_flags():
    flags.DEFINE_bool('lower_case', None, '')
    flags.DEFINE_integer('max_preds_per_seq', None, '')
    flags.DEFINE_string('bert_config', None, '')
    flags.DEFINE_string('init_checkpoint', None, '')


def main(_):
    ps = load_params()
    # tf.autograph.set_verbosity(1)
    # print(tf.autograph.to_code(Trafo.embed.python_function))
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    from absl import app, logging
    logging.set_verbosity(logging.INFO)  # DEBUG
    load_flags()
    app.run(main)
