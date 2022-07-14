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
# fine-tune seq2seq models for question answering

import logging
import random

from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.trainer_utils import EvalPrediction

from .params import TRAIN, EVAL, TEST, ALL, EACH
from .runner import Runner as Base
from .qa import Runner as QA

log = logging.getLogger(__name__)


NAMES = {
    "squad_v2": ("question", "context", "answer"),
}


class Runner(Base):
    AutoModel = AutoModelForSeq2SeqLM

    @property
    def cols(self):
        if self._cols is None:
            ps = self.params
            if ps.do_train:
                cs = self.dataset[TRAIN].column_names
            elif ps.do_eval:
                cs = self.dataset[EVAL].column_names
            elif ps.do_test:
                cs = self.dataset[TEST].column_names
            else:
                raise ValueError("There is nothing to do")
            ns = NAMES.get(ps.dataset_name, None)
            if ps.question_column is None:
                q = ns[0] if ns is not None else cs[0]
            else:
                q = ps.question_column
                if q not in cs:
                    raise ValueError(f"--question_column' needs to be in: {', '.join(cs)}")
            if ps.context_column is None:
                c = ns[1] if ns is not None else cs[1]
            else:
                c = ps.context_column
                if c not in cs:
                    raise ValueError(f"--context_column' needs to be in: {', '.join(cs)}")
            if ps.answer_column is None:
                a = ns[2] if ns is not None else cs[2]
            else:
                a = ps.answer_column
                if a not in cs:
                    raise ValueError(f"--answer_column' needs to be in: {', '.join(cs)}")
            self._cols = {ALL: cs, EACH: [q, c, a]}
        return self._cols

    @property
    def config(self):
        if self._config is None:
            ps = self.params
            x = ps.config_name if ps.config_name else ps.model_name
            if not x:
                raise ValueError("Config from scratch is not supported")
            if x:
                y = self.AutoConfig.from_pretrained(
                    x,
                    cache_dir=ps.cache_dir,
                    revision=ps.model_version,
                    use_auth_token=True if ps.use_auth_token else None,
                )
            self._config = y
        return self._config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            ps = self.params
            x = ps.tokenizer_name if ps.tokenizer_name else ps.model_name
            if not x:
                raise ValueError("Tokenizer from scratch is not supported")
            y = self.AutoTokenizer.from_pretrained(
                x,
                cache_dir=ps.cache_dir,
                use_fast=True,
                revision=ps.model_version,
                use_auth_token=True if ps.use_auth_token else None,
            )
            self._tokenizer = y
            if ps.max_seq_length > y.model_max_length:
                log.warning(f"Using max_seq_length={y.model_max_length}")
            self.max_seq_length = min(ps.max_seq_length, y.model_max_length)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            ps = self.params
            if ps.model_name:
                y = self.AutoModel.from_pretrained(
                    ps.model_name,
                    from_tf=bool(".ckpt" in ps.model_name),
                    config=self.config,
                    cache_dir=ps.cache_dir,
                    revision=ps.model_version,
                    use_auth_token=True if ps.use_auth_token else None,
                )
            else:
                log.info("Training new model")
                y = self.AutoModel.from_config(self.config)
            self._model = y
            if y.config.dec_START is None:
                raise ValueError("Needs `config.dec_START`")
            if ps.label_smoothing_factor > 0 and not hasattr(
                y, "prepare_decoder_input_ids_from_labels"
            ):
                log.warning("Needs `prepare_decoder_input_ids_from_labels` method for model")
        return self._model

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
        ps, t = self.params, self.tokenizer
        ins, ans = self.prep_batch(xs)
        y = t(ins, max_len=self.max_seq_length, padding=self.padding, truncation=True)
        with t.as_target_tokenizer():
            ls = t(ans, max_len=ps.max_answer_length, padding=self.padding, truncation=True)
        if self.padding == "max_len" and ps.ignore_pad_token_for_loss:
            ls["input_ids"] = [[(x if x != t.PAD else -100) for x in l] for l in ls["input_ids"]]
        y["labels"] = ls["input_ids"]
        return y

    def prep_batch(self, xs):
        q, c, a = self.cols[EACH]
        ins = [
            " ".join(["question:", q.lstrip(), "context:", c.lstrip()])
            for q, c in zip(xs[q], xs[c])
        ]
        ans = [a["text"][0] if len(a["text"]) > 0 else "" for a in xs[a]]
        return ins, ans

    @property
    def eval_ds(self):
        if self._eval_ds is None:
            ps, mgr = self.params, self.mgr
            y = super().eval_ds
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
        ps, t = self.params, self.tokenizer
        ins, ans = self.prep_batch(xs)
        y = t(
            ins,
            max_len=self.max_seq_length,
            padding=self.padding,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        with t.as_target_tokenizer():
            ls = t(ans, max_len=ps.max_answer_length, padding=self.padding, truncation=True)
        map = y.pop("overflow_to_sample_mapping")
        y["example_id"] = []
        for i in range(len(y["input_ids"])):
            y["example_id"].append(xs["id"][map[i]])
        if self.padding == "max_len" and ps.ignore_pad_token_for_loss:
            ls["input_ids"] = [[(x if x != t.PAD else -100) for x in l] for l in ls["input_ids"]]
        y["labels"] = ls["input_ids"]
        return y

    @property
    def test_ds(self):
        if self._test_ds is None:
            ps, mgr = self.params, self.mgr
            y = super().test_ds
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
    def metric(self):
        if self._metric is None:
            self.metric = load_metric("squad_v2" if self.ps.version_2_with_negative else "squad")
        return self._metric

    def compute_metrics(self, p):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    @property
    def loaders(self):
        if self._loaders is None:
            ps, t = self.params, self.tokenizer
            c = DataCollatorForSeq2Seq(
                t,
                model=self.model,
                label_pad_token_id=-100 if ps.ignore_pad_token_for_loss else t.PAD,
                pad_to_multiple_of=8 if ps.fp16 else None,
            )
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
        return self._loaders

    def post_proc(self, xs, features, outs, stage="eval"):
        ps = self.params
        preds = outs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        map = {k: i for i, k in enumerate(xs["id"])}
        feature_per_example = {map[x["example_id"]]: i for i, x in enumerate(features)}
        ys = {}
        for i, x in enumerate(xs):
            ys[x["id"]] = preds[feature_per_example[i]]
        if ps.version_2_with_negative:
            ys = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in ys.items()
            ]
        else:
            ys = [{"id": k, "prediction_text": v} for k, v in ys.items()]
        ls = [{"id": x["id"], "answers": x[self.cols[EACH][2]]} for x in xs]
        return EvalPrediction(predictions=ys, label_ids=ls)


def main():
    x = Runner()
    x.cols
    x.dataset
    x.config
    x.tokenizer
    x.model
    x.model.resize_token_embeddings(len(x.tokenizer))
    x.loaders
    x.prepare()
    x.train()
    x.eval()
    x.test()
    x.save()


if __name__ == "__main__":
    main()
