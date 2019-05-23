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

import json
import math

import pathlib as pth
import collections as co

from datetime import datetime

import tensorflow as tf

from google.protobuf import struct_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams

from qnarre.neura import bert
from qnarre.neura.layers import Squad
from qnarre.feeds.dset.squad_ds import dataset as squad_ds

ks = tf.keras
kls = ks.layers
kcb = ks.callbacks


def model_for(params):
    PS = params
    sh = (PS.max_seq_len, )
    toks = kls.Input(shape=sh, dtype='int32', name='tokens')
    segs = kls.Input(shape=sh, dtype='int32', name='segments')
    opts = kls.Input(shape=sh, dtype='int32', name='optimals')
    spans = kls.Input(shape=(2, ), dtype='int32', name='optimals')
    ins = [toks, segs, opts, spans]
    y = Squad(PS)(ins)
    return ks.Model(inputs=ins, outputs=[y])


def dataset_for(kind, params):
    ds = squad_ds(kind, params)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(params.batch_size)
        # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds
        ds = ds.batch(params.batch_size)
    return ds


RawResult = co.namedtuple('RawResult',
                          ['unique_id', 'start_logits', 'end_logits'])


def write_predictions(PS, all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    tf.logging.info('Writing predictions to: %s' % (output_prediction_file))
    tf.logging.info('Writing nbest to: %s' % (output_nbest_file))

    example_index_to_features = co.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = co.namedtuple(  # pylint: disable=invalid-name
        'PrelimPrediction', [
            'feature_index', 'start_index', 'end_index', 'start_logit',
            'end_logit'
        ])

    all_predictions = co.OrderedDict()
    all_nbest_json = co.OrderedDict()
    scores_diff_json = co.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if PS.version_2_with_negative:
                feature_null_score = result.start_logits[
                    0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if PS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = co.namedtuple(  # pylint: disable=invalid-name
            'NbestPrediction', ['text', 'start_logit', 'end_logit'])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = ' '.join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(' ##', '')
                tok_text = tok_text.replace('##', '')

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = ' '.join(tok_text.split())
                orig_text = ' '.join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ''
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if PS.version_2_with_negative:
            if '' not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text='',
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = co.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_logit'] = entry.start_logit
            output['end_logit'] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not PS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]['text']
        else:
            # predict '' iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > PS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ''
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')

    with tf.gfile.GFile(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')

    if PS.version_2_with_negative:
        with tf.gfile.GFile(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')


def get_final_text(PS, pred_text, orig_text, do_lower_case):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = co.OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = ' '.join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if PS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if PS.verbose_logging:
            tf.logging.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if PS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if PS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def run_squad(sess, params):
    # with tf.distribute.MirroredStrategy().scope():
    model = model_for(params)
    model.compile(
        optimizer=params.optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    ds_train = dataset_for('train', params)
    ds_test = dataset_for('test', params)

    save_p = pth.Path(params.dir_save)
    if save_p.exists():
        model.train_on_batch(ds_train[:1])
        model.load_weights(save_p)

    model.summary()

    p = params.log_dir + '/train/' + sess
    writer = tf.summary.create_file_writer(p)
    sum_s = hparams.session_start_pb(hparams=params.hparams)

    cbacks = [
        kcb.TensorBoard(
            log_dir=p,
            histogram_freq=1,
            embeddings_freq=0,
            update_freq='epoch'),
        # kcb.EarlyStopping(
        #     monitor='val_loss', min_delta=1e-2, patience=2, verbose=True),
    ]

    if save_p.exists():
        cbacks.append(
            kcb.ModelCheckpoint(
                model_save_path=save_p,
                save_best_only=True,
                monitor='val_loss',
                verbose=True))

    hist = model.fit(
        ds_train,
        callbacks=cbacks,
        epochs=params.train_epochs,
        validation_data=ds_test)
    print(f'History: {hist.history}')

    if save_p.exists():
        model.save_weights(save_p, save_format='tf')

    loss, acc = model.evaluate(ds_test)
    print(f'\nTest loss, acc: {loss}, {acc}')

    with writer.as_default():
        e = tf.compat.v1.Event(summary=sum_s).SerializeToString()
        tf.summary.import_event(e)
        tf.summary.scalar('accuracy', acc, step=1, description='Accuracy')
        sum_e = hparams.session_end_pb(api_pb2.STATUS_SUCCESS)
        e = tf.compat.v1.Event(summary=sum_e).SerializeToString()
        tf.summary.import_event(e)


_params = dict(
    seq_stride=128,
    max_ans_len=30,
    max_qry_len=64,
    max_seq_len=384,
    n_best_size=20,
    null_score_diff_threshold=0.0,
    train_epochs=2.0,
    use_fp16=False,
    use_xla=False,
    warmup_split=0.1,
    batch_size=8,
    learn_rate=5e-6,
)

_params.update(
    dir_data='.data/squad',
    log_dir='.model/squad/logs',
    dir_model='.model/squad',
    dir_save='.model/squad/save',
)


def main(_):
    PS = bert.load_params().override(_params)
    nus = [16, 32, 512]
    drs = [0.1, 0.2]
    writer = tf.summary.create_file_writer(PS.log_dir + '/train')
    with writer.as_default():
        s = _to_summary_pb(nus, drs, '')
        e = tf.compat.v1.Event(summary=s).SerializeToString()
        tf.summary.import_event(e)
    for nu in nus:
        for dr in drs:
            kw = {'num_units': nu, 'dropout_rate': dr}
            sess = datetime.now().strftime('%Y%m%d-%H%M%S')
            print(f'--- Running session {sess}:', kw)
            PS.update(**kw)
            run_squad(sess, PS)


def _to_summary_pb(num_units_list, dropout_rate_list, optimizer_list):
    nus_val = struct_pb2.ListValue()
    nus_val.extend(num_units_list)
    drs_val = struct_pb2.ListValue()
    drs_val.extend(dropout_rate_list)
    opts_val = struct_pb2.ListValue()
    opts_val.extend(optimizer_list)
    return hparams.experiment_pb(
        hparam_infos=[
            api_pb2.HParamInfo(
                name='num_units',
                display_name='Number of units',
                type=api_pb2.DATA_TYPE_FLOAT64,
                domain_discrete=nus_val),
            api_pb2.HParamInfo(
                name='dropout_rate',
                display_name='Dropout rate',
                type=api_pb2.DATA_TYPE_FLOAT64,
                domain_discrete=drs_val),
            api_pb2.HParamInfo(
                name='optimizer',
                display_name='Optimizer',
                type=api_pb2.DATA_TYPE_STRING,
                domain_discrete=opts_val)
        ],
        metric_infos=[
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag='accuracy'),
                display_name='Accuracy'),
        ])


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
'''
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /results/predictions.json
'''
