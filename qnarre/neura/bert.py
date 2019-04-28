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
# https://arxiv.org/pdf/1810.04805.pdf
# https://github.com/google-research/bert

# import pathlib as pth
from datetime import datetime

import tensorflow as T

from qnarre.neura.layers import Bert
from qnarre.neura import transformer, utils
from qnarre.feeds.prep.tokenizer import Tokenizer
from qnarre.feeds.dset.bert_ds import dataset as bert_ds

KS = T.keras
KL = KS.layers

# KC = KS.callbacks


def model_for(params):
    PS = params
    sh = (PS.max_seq_len, )
    seq = KL.Input(shape=sh, dtype='int32', name='seq')
    typ = KL.Input(shape=sh, dtype='int32', name='typ')
    sh = (PS.max_seq_preds, )
    idx = KL.Input(shape=sh, dtype='int32', name='mlm_idx')
    val = KL.Input(shape=sh, dtype='int32', name='mlm_val')
    fit = KL.Input(shape=sh, dtype='bool', name='fit')
    mlm = KL.Input(shape=sh, dtype='float32', name='mlm')
    ins = [seq, typ, fit, idx, val, mlm]
    outs = Bert(PS)(ins)
    m = KS.Model(inputs=ins, outputs=outs)
    m.compile(
        optimizer=utils.adam_opt(PS),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return m


def dset_for(kind, params):
    PS = params
    ds, data = bert_ds(kind, PS)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=T.data.experimental.AUTOTUNE)
    return ds, data


_params = dict(
    attn_drop=0.1,
    attn_heads=12,  # bert 12
    attn_k_size=0,
    attn_v_size=0,
    batch_size=32,
    checkpoint_steps=1000,
    decode_layers=0,
    dupe_factor=10,
    embed_drop=0.6,
    encode_layers=0,
    eval_batch_size=8,
    eval_steps=100,
    ffn_act='gelu',
    ffn_drop=0.2,
    ffn_units=3072,
    hidden_act='gelu',
    hidden_drop=0.1,
    hidden_size=768,  # 512
    init_checkpoint=None,
    init_stddev=0.02,  # stdev truncated_normal for all weights
    iters_per_loop=1000,
    l2_penalty=None,  # 1e-6, 1e-4
    learn_rate=5e-5,
    lower_case=None,
    max_pos_len=512,
    max_seq_preds=20,
    max_seq_len=128,
    token_types=16,
    param_attn_k_size=0,
    param_attn_v_size=0,
    pos_embed='timing',  # embed
    post_drop=0.1,
    prepost_drop=0.1,
    random_seed=12345,
    stack_layers=12,
    symbol_drop=0.0,
    train_steps=100000,
    vocab_size=None,
    warmup_steps=10000,
)

_params.update(
    data_dir='.data/bert',
    log_dir='.model/bert/logs',
    model_dir='.model/bert',
    save_dir='.model/bert/save',
    model_name='uncased_L-12_H-768_A-12',
)


def load_flags():
    transformer.load_flags()
    from absl import flags
    flags.DEFINE_bool('lower_case', None, '')
    flags.DEFINE_integer('max_preds_per_seq', None, '')
    flags.DEFINE_string('bert_config', None, '')
    flags.DEFINE_string('init_checkpoint', None, '')


def load_params():
    PS = transformer.load_params().override(_params)
    return PS.update(tokenizer=Tokenizer(PS))


def main(_):
    # bert_config = modeling.BertConfig.from_json_file(PS.bert_config)
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
    PS = load_params().override(_params)
    utils.train_sess(sid, PS, model_for, dset_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    load_flags()
    from absl import app
    app.run(main)

###
"""
def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
              val, masked_lm_weights,
              next_sentence_example_loss, next_sentence_log_probs,
              fit):
    masked_lm_log_probs = T.reshape(
        masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = T.argmax(
        masked_lm_log_probs, axis=-1, output_type=T.int32)
    masked_lm_example_loss = T.reshape(masked_lm_example_loss,
                                        [-1])
    val = T.reshape(val, [-1])
    masked_lm_weights = T.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = T.metrics.accuracy(
        labels=val,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = T.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = T.reshape(
        next_sentence_log_probs,
        [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = T.argmax(
        next_sentence_log_probs, axis=-1, output_type=T.int32)
    fit = T.reshape(fit, [-1])
    next_sentence_accuracy = T.metrics.accuracy(
        labels=fit,
        predictions=next_sentence_predictions)
    next_sentence_mean_loss = T.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }
"""
