# Copyright 2021 Quantapix Authors. All Rights Reserved.
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
# fine-tune XLNet for question answering with beam search

import collections
import json
import logging
import numpy as np
import os
import torch

from tqdm.auto import tqdm
from transformers import (
    XLNetConfig,
    XLNetTokenizerFast,
    XLNetForQuestionAnswering,
    EvalPrediction,
)

from .params import EVAL, TEST, EACH
from .qa import Runner as Base
from .utils import init_array

log = logging.getLogger(__name__)


class Runner(Base):
    AutoConfig = XLNetConfig
    AutoTokenizer = XLNetTokenizerFast
    AutoModel = XLNetForQuestionAnswering

    def prep_for_train(self, xs):
        ps, pad_on_right = self.params, self.pad_on_right
        q_col, c_col, a_col = self.cols[EACH]
        xs[q_col] = [x.lstrip() for x in xs[q_col]]
        ys = self.tokenizer(
            xs[q_col if pad_on_right else c_col],
            xs[c_col if pad_on_right else q_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_len=self.max_seq_length,
            stride=ps.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_len",
        )
        map = ys.pop("overflow_to_sample_mapping")
        specials = ys.pop("special_tokens_mask")
        ys["start_positions"] = []
        ys["end_positions"] = []
        ys["is_impossible"] = []
        ys["cls_index"] = []
        ys["p_mask"] = []
        for i, offs in enumerate(ys.pop("offset_mapping")):
            ins = ys["input_ids"][i]
            cls = ins.index(self.tokenizer.cls_token_id)
            ys["cls_index"].append(cls)
            ids = ys["typ_ids"][i]
            for k, s in enumerate(specials[i]):
                if s:
                    ids[k] = 3
            por = 1 if pad_on_right else 0
            ys["p_mask"].append(
                [
                    0.0 if (not specials[i][k] and s == por) or k == cls else 1.0
                    for k, s in enumerate(ids)
                ]
            )
            ans = xs[a_col][map[i]]
            if len(ans["answer_start"]) == 0:
                ys["start_positions"].append(cls)
                ys["end_positions"].append(cls)
                ys["is_impossible"].append(1.0)
            else:
                s = ans["answer_start"][0]
                e = s + len(ans["text"][0])
                j = 0
                while ids[j] != por:
                    j += 1
                k = len(ins) - 1
                while ids[k] != por:
                    k -= 1
                if not (offs[j][0] <= s and offs[k][1] >= e):
                    ys["start_positions"].append(cls)
                    ys["end_positions"].append(cls)
                    ys["is_impossible"].append(1.0)
                else:
                    while j < len(offs) and offs[j][0] <= s:
                        j += 1
                    ys["start_positions"].append(j - 1)
                    while offs[k][1] >= e:
                        k -= 1
                    ys["end_positions"].append(k + 1)
                    ys["is_impossible"].append(0.0)
        return ys

    def prep_for_eval(self, xs):
        ps, pad_on_right = self.params, self.pad_on_right
        q_col, c_col, _ = self.cols[EACH]
        xs[q_col] = [q.lstrip() for q in xs[q_col]]
        ys = self.tokenizer(
            xs[q_col if pad_on_right else c_col],
            xs[c_col if pad_on_right else q_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_len=self.max_seq_length,
            stride=ps.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_len",
        )
        map = ys.pop("overflow_to_sample_mapping")
        specials = ys.pop("special_tokens_mask")
        ys["example_id"] = []
        ys["cls_index"] = []
        ys["p_mask"] = []
        for i, ins in enumerate(ys["input_ids"]):
            cls = ins.index(self.tokenizer.cls_token_id)
            ys["cls_index"].append(cls)
            ids = ys["typ_ids"][i]
            for k, s in enumerate(specials[i]):
                if s:
                    ids[k] = 3
            por = 1 if pad_on_right else 0
            ys["p_mask"].append(
                [
                    0.0 if (not specials[i][k] and s == por) or k == cls else 1.0
                    for k, s in enumerate(ids)
                ]
            )
            ys["example_id"].append(xs["id"][map[i]])
            ys["offset_mapping"][i] = [
                (o if ids[k] == por else None) for k, o in enumerate(ys["offset_mapping"][i])
            ]
        return ys

    def eval(self):
        ps, mgr, ds = self.params, self.mgr
        ds, src = self.eval_ds, self.loaders[EVAL]
        log.info("*** Evaluating ***")
        log.info(f"  Num samples = {len(ds)}")
        log.info(f"  Batch size per device = {ps.eval_batch_size}")

        all_start_top_log_probs = []
        all_start_top_index = []
        all_end_top_log_probs = []
        all_end_top_index = []
        all_cls_logits = []
        for xs in src:
            with torch.no_grad():
                ys = self.model(**xs)
                start_top_log_probs = ys.start_top_log_probs
                start_top_index = ys.start_top_index
                end_top_log_probs = ys.end_top_log_probs
                end_top_index = ys.end_top_index
                cls_logits = ys.cls_logits

                if not ps.pad_to_max_length:
                    start_top_log_probs = mgr.pad_across_processes(
                        start_top_log_probs, dim=1, PAD=-100
                    )
                    start_top_index = mgr.pad_across_processes(start_top_index, dim=1, PAD=-100)
                    end_top_log_probs = mgr.pad_across_processes(end_top_log_probs, dim=1, PAD=-100)
                    end_top_index = mgr.pad_across_processes(end_top_index, dim=1, PAD=-100)
                    cls_logits = mgr.pad_across_processes(cls_logits, dim=1, PAD=-100)

                all_start_top_log_probs.append(mgr.gather(start_top_log_probs).cpu().numpy())
                all_start_top_index.append(mgr.gather(start_top_index).cpu().numpy())
                all_end_top_log_probs.append(mgr.gather(end_top_log_probs).cpu().numpy())
                all_end_top_index.append(mgr.gather(end_top_index).cpu().numpy())
                all_cls_logits.append(mgr.gather(cls_logits).cpu().numpy())

        l = max([x.shape[1] for x in all_end_top_log_probs])
        start_top_log_probs_concat = init_array(all_start_top_log_probs, ds, l)
        start_top_index_concat = init_array(all_start_top_index, ds, l)
        end_top_log_probs_concat = init_array(all_end_top_log_probs, ds, l)
        end_top_index_concat = init_array(all_end_top_index, ds, l)

        cls_logits_concat = np.concatenate(all_cls_logits, axis=0)

        del start_top_log_probs
        del start_top_index
        del end_top_log_probs
        del end_top_index
        del cls_logits

        outputs_numpy = (
            start_top_log_probs_concat,
            start_top_index_concat,
            end_top_log_probs_concat,
            end_top_index_concat,
            cls_logits_concat,
        )
        y = self.post_proc(self.evals, ds, outputs_numpy)
        y = self.metric.compute(predictions=y.predictions, references=y.label_ids)
        log.info(f"Evaluation metrics: {y}")

    def post_proc(self, xs, features, preds, stage="eval"):
        ps = self.params
        ys, diff = proc_preds(
            examples=xs,
            features=features,
            predictions=preds,
            version_2_with_negative=ps.version_2_with_negative,
            n_best_size=ps.n_best_size,
            max_answer_length=ps.max_answer_length,
            start_n_top=self.model.config.start_n_top,
            end_n_top=self.model.config.end_n_top,
            out_dir=ps.out_dir,
            prefix=stage,
        )
        if ps.version_2_with_negative:
            ys = [
                {"id": k, "prediction_text": v, "no_answer_probability": diff[k]}
                for k, v in ys.items()
            ]
        else:
            ys = [{"id": k, "prediction_text": v} for k, v in ys.items()]
        ids = [{"id": x["id"], "answers": x[self.cols[EACH][2]]} for x in xs]
        return EvalPrediction(predictions=ys, label_ids=ids)

    def pred(self):
        ps, mgr = self.params, self.mgr
        if ps.do_test:
            ds, src = self.test_ds, self.loaders[TEST]
            log.info("*** Prediction ***")
            log.info(f"  Num samples = {len(ds)}")
            log.info(f"  Batch size per device = {ps.eval_batch_size}")
            all_start_top_log_probs = []
            all_start_top_index = []
            all_end_top_log_probs = []
            all_end_top_index = []
            all_cls_logits = []
            for xs in src:
                with torch.no_grad():
                    ys = self.model(**xs)
                    start_top_log_probs = ys.start_top_log_probs
                    start_top_index = ys.start_top_index
                    end_top_log_probs = ys.end_top_log_probs
                    end_top_index = ys.end_top_index
                    cls_logits = ys.cls_logits

                    if not ps.pad_to_max_length:
                        start_top_log_probs = mgr.pad_across_processes(
                            start_top_log_probs, dim=1, PAD=-100
                        )
                        start_top_index = mgr.pad_across_processes(start_top_index, dim=1, PAD=-100)
                        end_top_log_probs = mgr.pad_across_processes(
                            end_top_log_probs, dim=1, PAD=-100
                        )
                        end_top_index = mgr.pad_across_processes(end_top_index, dim=1, PAD=-100)
                        cls_logits = mgr.pad_across_processes(cls_logits, dim=1, PAD=-100)

                    all_start_top_log_probs.append(mgr.gather(start_top_log_probs).cpu().numpy())
                    all_start_top_index.append(mgr.gather(start_top_index).cpu().numpy())
                    all_end_top_log_probs.append(mgr.gather(end_top_log_probs).cpu().numpy())
                    all_end_top_index.append(mgr.gather(end_top_index).cpu().numpy())
                    all_cls_logits.append(mgr.gather(cls_logits).cpu().numpy())

            l = max([x.shape[1] for x in all_end_top_log_probs])

            start_top_log_probs_concat = init_array(all_start_top_log_probs, ds, l)
            start_top_index_concat = init_array(all_start_top_index, ds, l)
            end_top_log_probs_concat = init_array(all_end_top_log_probs, ds, l)
            end_top_index_concat = init_array(all_end_top_index, ds, l)
            cls_logits_concat = np.concatenate(all_cls_logits, axis=0)

            del start_top_log_probs
            del start_top_index
            del end_top_log_probs
            del end_top_index
            del cls_logits

            outputs_numpy = (
                start_top_log_probs_concat,
                start_top_index_concat,
                end_top_log_probs_concat,
                end_top_index_concat,
                cls_logits_concat,
            )

            y = self.post_proc(self.preds, ds, outputs_numpy)
            y = self.metric.compute(predictions=y.predictions, references=y.label_ids)
            log.info(f"Prediction metrics: {y}")


def proc_preds(
    examples,
    features,
    predictions,
    version_2_with_negative=False,
    n_best_size=20,
    max_answer_length=30,
    start_n_top=5,
    end_n_top=5,
    out_dir=None,
    prefix=None,
    log_level=logging.WARNING,
):
    if len(predictions) != 5:
        raise ValueError("`predictions` should be a tuple with five elements.")
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions
    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None
    log.setLevel(log_level)
    log.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(
                        str(start_index), False
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size
        ]
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]
        if len(predictions) == 0:
            predictions.insert(
                0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6}
            )
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)
        all_nbest_json[example["id"]] = [
            {
                k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
                for k, v in pred.items()
            }
            for pred in predictions
        ]
    if out_dir is not None:
        if not os.path.isdir(out_dir):
            raise EnvironmentError(f"{out_dir} is not a directory.")
        prediction_file = os.path.join(
            out_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            out_dir,
            "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json",
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                out_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )
        log.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        log.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            log.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    return all_predictions, scores_diff_json


def main():
    x = Runner()
    x.dataset
    x.cols
    x.config
    x.tokenizer
    x.model
    x.loaders
    x.prepare()
    x.train()
    x.eval()
    x.pred()
    x.save()


if __name__ == "__main__":
    main()
