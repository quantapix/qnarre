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
# https://arxiv.org/pdf/1706.03762.pdf
# https://github.com/tensorflow/tensor2tensor

import tensorflow as T

from qnarre.neura import utils as U
from qnarre.neura.layers import Transformer
from qnarre.feeds.dset.squad_ds import dataset as squad_ds

KS = T.keras
KL = KS.layers


def model_for(params):
    PS = params
    sh = (PS.max_seq_len, )
    src = KS.Input(shape=sh, dtype='int32', name='src')
    tgt = KS.Input(shape=sh, dtype='int32', name='tgt')
    ins = [[src, None], [tgt, None]]
    outs = Transformer(PS)(ins)
    m = KS.Model(inputs=ins, outputs=outs)
    m.compile(optimizer=U.adam_opt(PS),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def dset_for(kind, params):
    PS = params
    ds = squad_ds(kind, PS)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


_params = dict(
    attn_heads=8,
    attn_k_size=64,
    attn_type='dot_attn',
    attn_v_size=64,
    batch_size=8,
    causal_refl=None,
    decode_layers=0,
    encode_layers=0,
    ffn_type='dense_dense',
    ffn_units=2048,
    hidden_drop=0.1,
    hidden_size=512,
    learn_rate=5e-6,
    max_ans_len=30,
    max_qry_len=64,
    max_seq_len=384,
    n_best_size=20,
    null_score_diff_threshold=0.0,
    prox_bias=None,
    refl_type='dot_attn',
    seq_stride=128,
    stack_layers=6,
    train_epochs=2.0,
    use_fp16=False,
    use_xla=False,
    vocab_size=None,
    warmup_split=0.1,
    prepost_bdims='',
    norm_epsilon=1e-6,
    post_cmd='dan',
    pre_cmd='n',
    norm_type='layer',
)

_params.update(
    dir_data='.data/transformer',
    log_dir='.model/transformer/logs',
    dir_model='.model/transformer',
    dir_save='.model/transformer/save',
)


def load_params():
    f = 'channels_first' if T.test.is_built_with_cuda() else 'channels_last'
    return U.Params(_params, data_format=f)


def load_flags():
    from absl import flags
    flags.DEFINE_bool('xyz', None, '')


def main(_):
    U.train_sess(load_params(), model_for, dset_for)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    load_flags()
    from absl import app
    app.run(main)
