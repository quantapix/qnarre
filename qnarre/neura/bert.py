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

from qnarre.neura import trafo
from qnarre.neura.session import session_for

from qnarre.feeds.dset.squad import dset as bert_dset
from qnarre.neura.layers.squad import model as bert_model
from qnarre.neura.layers.squad import adapter as bert_adapter


def dset_for(ps, kind):
    ds, feats = bert_dset(ps, kind)
    if kind == 'train':
        ds = ds.shuffle(10000)
    ds = ds.batch(1 if ps.eager_mode else ps.batch_size)
    ds = ds.map(lambda d: bert_adapter(ps, feats, d))
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(ps, compiled=False):
    m = bert_model(ps)
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
    model='uncased_L-12_H-768_A-12',
)


def load_params():
    return trafo.load_params().override(params)


def load_flags():
    trafo.load_flags()
    from absl import flags
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
    from absl import app, flags, logging
    logging.set_verbosity(logging.INFO)  # DEBUG
    load_flags()
    flags.DEFINE_integer('xxx', None, '')
    app.run(main)
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
