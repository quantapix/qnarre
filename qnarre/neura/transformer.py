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

from datetime import datetime

import tensorflow as tf

from qnarre.neura import bert, utils
from qnarre.neura.layers import Transformer
from qnarre.feeds.dset.squad_ds import dataset as squad_ds

ks = tf.keras
kls = ks.layers

# kcb = ks.callbacks


def model_for(params):
    PS = params
    sh = (PS.max_seq_len, )
    src = kls.Input(shape=sh, dtype='int32', name='src')
    tgt = kls.Input(shape=sh, dtype='int32', name='tgt')
    ins = [[src, None], [tgt, None]]
    outs = Transformer(PS)(ins)
    m = ks.Model(inputs=ins, outputs=outs)
    m.compile(
        optimizer=utils.adam_opt(PS),
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
    batch_size=8,
    ffn_units=2048,
    hidden_size=512,
    learn_rate=5e-6,
    max_ans_len=30,
    max_qry_len=64,
    max_seq_len=384,
    n_best_size=20,
    null_score_diff_threshold=0.0,
    seq_stride=128,
    train_epochs=2.0,
    use_fp16=False,
    use_xla=False,
    vocab_size=None,
    warmup_split=0.1,
    hidden_drop=0.1,
    decode_layers=0,
    encode_layers=0,
    stack_layers=6,
    attn_k_size=64,
    attn_v_size=64,
    attn_heads=8,
)

_params.update(
    data_dir='.data/transformer',
    log_dir='.model/transformer/logs',
    model_dir='.model/transformer',
    save_dir='.model/transformer/save',
)


def main(_):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
    PS = bert.load_params().override(_params)
    utils.train_sess(sid, PS, model_for, dset_for)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    bert.load_flags()
    from absl import flags
    flags.DEFINE_float('null_score_diff_threshold', None, '')
    flags.DEFINE_float('warmup_split', None, '')
    flags.DEFINE_integer('max_ans_len', None, '')
    flags.DEFINE_integer('max_qry_len', None, '')
    flags.DEFINE_integer('n_best_size', None, '')
    flags.DEFINE_integer('seq_stride', None, '')
    from absl import app
    app.run(main)
