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
# fine-tune for question answering

import collections
import json
import logging
import numpy as np
import os
import random
import torch

from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    AutoModelForQuestionAnswering,
    EvalPrediction,
)

from ..params import TRAIN, EVAL, TEST, ALL, EACH
from ..runner import Runner as Base
from ..utils import init_array

log = logging.getLogger(__name__)


class Runner(Base):
    AutoModel = AutoModelForQuestionAnswering

    @property
    def cols(self):
        if self._cols is None:
            cs = self.dataset[TRAIN].column_names
            q = "question" if "question" in cs else cs[0]
            c = "context" if "context" in cs else cs[1]
            a = "answers" if "answers" in cs else cs[2]
            self._cols = {ALL: cs, EACH: [q, c, a]}
        return self._cols

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            ps, t = self.params, super().tokenizer
            self.pad_on_right = t.padding_side == "right"
            if ps.max_seq_length > t.model_max_length:
                log.warning(f"Using max_seq_length={t.model_max_length}")
            self.max_seq_length = min(ps.max_seq_length, t.model_max_length)
        return self._tokenizer

    @property
    def train_ds(self):
        if self._train_ds is None:
            ps, mgr, ds = self.params, self.mgr, self.dataset
            y = ds[TRAIN]
            if ps.max_train_samples is not None:
                y = y.select(range(ps.max_train_samples))
            with mgr.main_process_first():
                y = y.map(
                    self.prep_for_train,
                    batched=True,
                    num_proc=ps.num_workers,
                    remove_columns=self.cols[ALL],
                    load_from_cache_file=not ps.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            for i in random.sample(range(len(y)), 3):
                log.info(f"Sample {i} of the training set: {y[i]}")
            self._train_ds = y
        return self._train_ds

    def prep_for_train(self, xs):
        ps, pad_on_right = self.params, self.pad_on_right
        q, c, a = self.cols[EACH]
        xs[q] = [x.lstrip() for x in xs[q]]
        ys = self.tokenizer(
            xs[q if pad_on_right else c],
            xs[c if pad_on_right else q],
            truncation="only_second" if pad_on_right else "only_first",
            max_len=self.max_seq_length,
            stride=ps.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )
        map = ys.pop("overflow_to_sample_mapping")
        ys["start_positions"] = []
        ys["end_positions"] = []
        for i, offs in enumerate(ys.pop("offset_mapping")):
            ins = ys["input_ids"][i]
            ids = ys.sequence_ids(i)
            ans = xs[a][map[i]]
            cls = ins.index(self.tokenizer.cls_token_id)
            por = 1 if pad_on_right else 0
            if len(ans["answer_start"]) == 0:
                ys["start_positions"].append(cls)
                ys["end_positions"].append(cls)
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
                else:
                    while j < len(offs) and offs[j][0] <= s:
                        j += 1
                    ys["start_positions"].append(j - 1)
                    while offs[k][1] >= e:
                        k -= 1
                    ys["end_positions"].append(k + 1)
        return ys

    @property
    def eval_ds(self):
        if self._eval_ds is None:
            ps, mgr = self.params, self.mgr
            self.evals = y = super().eval_ds
            with mgr.main_process_first():
                y = y.map(
                    self.prep_for_eval,
                    batched=True,
                    num_proc=ps.num_workers,
                    remove_columns=self.cols[ALL],
                    load_from_cache_file=not ps.overwrite_cache,
                    desc="Running tokenizer on eval dataset",
                )
            self._eval_ds = y
        return self._eval_ds

    def prep_for_eval(self, xs):
        ps, pad_on_right = self.params, self.pad_on_right
        q, c, _ = self.cols[EACH]
        xs[q] = [q.lstrip() for q in xs[q]]
        ys = self.tokenizer(
            xs[q if pad_on_right else c],
            xs[c if pad_on_right else q],
            truncation="only_second" if pad_on_right else "only_first",
            max_len=self.max_seq_length,
            stride=ps.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )
        map = ys.pop("overflow_to_sample_mapping")
        ys["example_id"] = []
        for i in range(len(ys["input_ids"])):
            ids = ys.sequence_ids(i)
            pi = 1 if pad_on_right else 0
            ys["example_id"].append(xs["id"][map[i]])
            ys["offset_mapping"][i] = [
                (v if ids[k] == pi else None) for k, v in enumerate(ys["offset_mapping"][i])
            ]
        return ys

    @property
    def test_ds(self):
        if self._test_ds is None:
            ps, mgr = self.params, self.mgr
            self.tests = y = super().test_ds
            with mgr.main_process_first():
                y = y.map(
                    self.prep_for_eval,
                    batched=True,
                    num_proc=ps.num_workers,
                    remove_columns=self.cols[ALL],
                    load_from_cache_file=not ps.overwrite_cache,
                    desc="Running tokenizer on test dataset",
                )
            self._test_ds = y
        return self._test_ds

    @property
    def loaders(self):
        if self._loaders is None:
            ps, mgr = self.params, self.mgr
            if ps.pad_to_max_length:
                c = default_data_collator
            else:
                c = DataCollatorWithPadding(
                    self.tokenizer, pad_to_multiple_of=(8 if mgr.use_fp16 else None)
                )
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            x = self.eval_ds.remove_columns(["example_id", "offset_mapping"])
            e = DataLoader(x, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
            if ps.do_test:
                x = self.test_ds.remove_columns(["example_id", "offset_mapping"])
                p = DataLoader(x, collate_fn=c, batch_size=ps.eval_batch_size)
                self._loaders[TEST] = p
        return self._loaders

    @property
    def metric(self):
        if self._metric is None:
            self._metric = load_metric("squad_v2" if self.ps.version_2_with_negative else "squad")
        return self._metric

    def prepare(self):
        mgr, ls = self.mgr, self.loaders
        t, e = ls[TRAIN], ls[EVAL]
        self._model, self._optimizer, t, e = mgr.prepare(self.model, self.optimizer, t, e)
        self._loaders = {TRAIN: t, EVAL: e, TEST: ls[TEST]}

    def eval(self):
        ps, mgr, ds = self.params, self.mgr
        ds, src = self.eval_ds, self.loaders[EVAL]
        log.info("*** Evaluating ***")
        log.info(f"  Num samples = {len(ds)}")
        log.info(f"  Batch size per device = {ps.eval_batch_size}")
        sss = []
        ess = []
        for xs in src:
            with torch.no_grad():
                ys = self.model(**xs)
                ss, es = ys.start_logits, ys.end_logits
                if not ps.pad_to_max_length:
                    ss = mgr.pad_across_processes(ss, dim=1, PAD=-100)
                    es = mgr.pad_across_processes(es, dim=1, PAD=-100)
                sss.append(mgr.gather(ss).cpu().numpy())
                ess.append(mgr.gather(es).cpu().numpy())
        l = max([x.shape[1] for x in sss])
        ss = init_array(sss, ds, l)
        es = init_array(ess, ds, l)
        del sss
        del ess
        y = self.post_proc(self.evals, ds, (ss, es))
        y = self.metric.compute(predictions=y.predictions, references=y.label_ids)
        log.info(f"Evaluation metrics: {y}")

    def post_proc(self, xs, features, preds, stage="eval"):
        ps = self.params
        ys = proc_tests(
            examples=xs,
            features=features,
            predictions=preds,
            version_2_with_negative=ps.version_2_with_negative,
            n_best_size=ps.n_best_size,
            max_answer_length=ps.max_answer_length,
            null_score_diff_threshold=ps.null_score_diff_threshold,
            out_dir=ps.out_dir,
            prefix=stage,
        )
        if ps.version_2_with_negative:
            ys = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in ys.items()
            ]
        else:
            ys = [{"id": k, "prediction_text": v} for k, v in ys.items()]
        ids = [{"id": x["id"], "answers": x[self.cols[EACH][2]]} for x in xs]
        return EvalPrediction(predictions=ys, label_ids=ids)

    def test(self):
        ps, mgr = self.params, self.mgr
        if ps.do_test:
            ds, src = self.test_ds, self.loaders[TEST]
            log.info("*** Prediction ***")
            log.info(f"  Num samples = {len(ds)}")
            log.info(f"  Batch size per device = {ps.eval_batch_size}")
            sss = []
            ess = []
            for xs in src:
                with torch.no_grad():
                    ys = self.model(**xs)
                    ss, es = ys.start_logits, ys.end_logits
                    if not ps.pad_to_max_length:
                        ss = mgr.pad_across_processes(ss, dim=1, PAD=-100)
                        es = mgr.pad_across_processes(ss, dim=1, PAD=-100)
                    sss.append(mgr.gather(ss).cpu().numpy())
                    ess.append(mgr.gather(es).cpu().numpy())
            l = max([x.shape[1] for x in sss])
            ss = init_array(sss, ds, l)
            es = init_array(ess, ds, l)
            del sss
            del ess
            y = self.post_proc(self.tests, ds, (ss, es))
            x = self.metric.compute(predictions=y.predictions, references=y.label_ids)
            log.info(f"Prediction metrics: {x}")


def proc_tests(
    examples,
    features,
    predictions,
    version_2_with_negative=False,
    n_best_size=20,
    max_answer_length=30,
    null_score_diff_threshold=0.0,
    out_dir=None,
    prefix=None,
    log_level=logging.WARNING,
):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions
    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()
    log.setLevel(log_level)
    log.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []
        for f in feature_indices:
            start_logits = all_start_logits[f]
            end_logits = all_end_logits[f]
            offset_mapping = features[f]["offset_mapping"]
            token_is_max_context = features[f].get("token_is_max_context", None)
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for i in start_indexes:
                for j in end_indexes:
                    if (
                        i >= len(offset_mapping)
                        or j >= len(offset_mapping)
                        or offset_mapping[i] is None
                        or offset_mapping[j] is None
                    ):
                        continue
                    if j < i or j - i + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(
                        str(i), False
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[i][0], offset_mapping[j][1]),
                            "score": start_logits[i] + end_logits[j],
                            "start_logit": start_logits[i],
                            "end_logit": end_logits[j],
                        }
                    )
        if version_2_with_negative:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size
        ]
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]
            score_diff = (
                null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]
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
    return all_predictions


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
    x.test()
    x.save()


if __name__ == "__main__":
    main()


"""
python qa.py \
  --model_name bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --out_dir ~/tmp/debug_squad

accelerate launch qa.py \
  --model_name bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --out_dir ~/tmp/debug_squad
"""
