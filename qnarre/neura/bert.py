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

# import pathlib as pth
from datetime import datetime

import tensorflow as tf

from qnarre.neura.layers import Bert
from qnarre.neura import transformer, utils
from qnarre.feeds.prep.tokenizer import Tokenizer
from qnarre.feeds.dset.bert_ds import dataset as bert_ds

ks = tf.keras
kls = ks.layers

# kcb = ks.callbacks


def model_for(params):
    PS = params
    sh = (PS.max_seq_len, )
    seq = kls.Input(shape=sh, dtype='int32', name='tokens')
    typ = kls.Input(shape=sh, dtype='int32', name='types')
    fit = kls.Input(shape=sh, dtype='int32', name='mlm_idxs')
    sh = (PS.max_seq_preds, )
    idx = kls.Input(shape=sh, dtype='int32', name='mlm_idxs')
    val = kls.Input(shape=sh, dtype='int32', name='mlm_vals')
    ws = kls.Input(shape=sh, dtype='float32', name='mlm_ws')
    ins = [seq, typ, fit, idx, val, ws]
    outs = Bert(PS)(ins)
    m = ks.Model(inputs=ins, outputs=outs)
    m.compile(
        optimizer=utils.adam_opt(PS),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return m


def dset_for(kind, params):
    PS = params
    ds = bert_ds(kind, PS)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_fn_builder(bert_config, init_checkpoint, learn_rate, train_steps,
                     warmup_steps, use_tpu, use_one_hot_embeddings):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(
                bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


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
    # tf.logging.set_verbosity(tf.logging.INFO)
    load_flags()
    from absl import app
    app.run(main)

###
"""
def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
              masked_lm_ids, masked_lm_weights,
              next_sentence_example_loss, next_sentence_log_probs,
              next_sentence_labels):
    masked_lm_log_probs = tf.reshape(
        masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                        [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs,
        [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(
        labels=next_sentence_labels,
        predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }
"""
