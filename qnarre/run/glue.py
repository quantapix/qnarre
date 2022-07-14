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
# fine-tune for sequence classification on GLUE

import logging
import random

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

from .params import TRAIN, EVAL, ALL, LABEL
from .runner import Runner as Base

log = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class Runner(Base):
    AutoModel = AutoModelForSequenceClassification

    @property
    def dataset(self):
        if self._dataset is None:
            ps = self.params
            if ps.task_name is not None:
                y = load_dataset("glue", ps.task_name)
            else:
                y = super().dataset
            self._dataset = y
        return self._dataset

    @property
    def cols(self):
        if self._cols is None:
            ps, ds = self.params, self.dataset
            cs = self.dataset[TRAIN].column_names
            self.labels = ("",)
            if ps.task_name is not None:
                self.is_regression = ps.task_name == "stsb"
                if not self.is_regression:
                    self.labels = ds[TRAIN].features[LABEL].names
                self.key1, self.key2 = task_to_keys[ps.task_name]
            else:
                self.is_regression = ds[TRAIN].features[LABEL].dtype in ["float32", "float64"]
                if not self.is_regression:
                    self.labels = ds[TRAIN].unique(LABEL).sort()
                xs = [x for x in cs if x != LABEL]
                if "sentence1" in xs and "sentence2" in xs:
                    self.key1, self.key2 = "sentence1", "sentence2"
                else:
                    if len(xs) >= 2:
                        self.key1, self.key2 = xs[:2]
                    else:
                        self.key1, self.key2 = xs[0], None
            self._cols = {ALL: cs}
        return self._cols

    @property
    def config(self):
        if self._config is None:
            ps = self.params
            self._config = AutoConfig.from_pretrained(
                ps.model_name, n_labels=len(self.labels), finetune=ps.task_name
            )
        return self._config

    @property
    def model(self):
        if self._model is None:
            ps, ls = self.params, self.labels
            m = super().model
            self.ids = None
            if (
                m.config.label2id != PretrainedConfig(n_labels=len(self.labels)).label2id
                and ps.task_name is not None
                and not self.is_regression
            ):
                ids = {l.lower(): i for l, i in m.config.label2id.items()}
                if list(sorted(ids.keys())) == list(sorted(ls)):
                    log.info(f"Using config label map: {ids}")
                    self.ids = {l(ids[l]) for l in ls}
                else:
                    log.warning(
                        f"Ignoring mismatched {list(sorted(ids.keys()))} vs {list(sorted(ls))}"
                    )
            elif ps.task_name is None:
                self.ids = {l: i for i, l in enumerate(ls)}
            if self.ids is not None:
                m.config.label2id = self.ids
                m.config.id2label = {i: l for l, i in self.config.label2id.items()}
            elif ps.task_name is not None and not self.is_regression:
                m.config.label2id = {l: i for i, l in enumerate(ls)}
                m.config.id2label = {i: l for l, i in self.config.label2id.items()}
        return self._model

    @property
    def train_ds(self):
        if self._train_ds is None:
            ps, mgr, ds = self.params, self.mgr, self.dataset
            with mgr.main_process_first():
                self._dataset = y = ds.map(
                    self.prep_for_train,
                    batched=True,
                    remove_columns=self.cols[ALL],
                    desc="Running tokenizer on dataset",
                )
            y = y[TRAIN]
            if ps.max_train_samples is not None:
                y = y.select(range(ps.max_train_samples))
            for i in random.sample(range(len(y)), 3):
                log.info(f"Sample {i} of the training set: {y[i]}")
            self._train_ds = y
        return self._train_ds

    def prep_for_train(self, xs):
        ps, k1, k2 = self.params, self.key1, self.key2
        texts = (xs[k1],) if k2 is None else (xs[k1], xs[k2])
        y = self.tokenizer(*texts, padding=self.padding, max_len=ps.max_len, truncation=True)
        if LABEL in xs:
            if self.ids is None:
                y["labels"] = xs[LABEL]
            else:
                y["labels"] = [self.ids[x] for x in xs[LABEL]]
        return y

    @property
    def eval_ds(self):
        if self._eval_ds is None:
            ps, ds = self.params, self.dataset
            y = ds["validation_matched" if ps.task_name == "mnli" else EVAL]
            if ps.max_eval_samples is not None:
                y = y.select(range(ps.max_eval_samples))
            self._eval_ds = y
        return self._eval_ds

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
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
        return self._loaders

    @property
    def metric(self):
        if self._metric is None:
            ps = self.params
            if ps.task_name is not None:
                y = load_metric("glue", ps.task_name)
            else:
                y = load_metric("accuracy")
            self._metric = y
        return self._metric

    def eval_epoch(self, e):
        m, mgr = self.model, self.mgr
        m.eval()
        for xs in self.loaders[EVAL]:
            ys = m(**xs)
            ys = ys.logits.argmax(dim=-1) if not self.is_regression else ys.logits.squeeze()
            self.metric.add_batch(predictions=mgr.gather(ys), references=mgr.gather(xs["labels"]))
        y = self.metric.compute()
        mgr.print(f"epoch {e}: {y}")

    def eval(self):
        ps, m, mgr = self.params, self.model, self.mgr
        if ps.task_name == "mnli":
            m.eval()
            for xs in self.loaders[EVAL]:
                ys = m(**xs)
                ys = ys.logits.argmax(dim=-1)
                self.metric.add_batch(
                    predictions=mgr.gather(ys), references=mgr.gather(xs["labels"])
                )
            y = self.metric.compute()
            mgr.print(f"mnli-mm: {y}")


def main():
    ps = [("--task_name", {"type", "default": None, "choices": list(task_to_keys.keys())})]
    x = Runner(ps)
    x.dataset
    x.config
    x.tokenizer
    x.model
    x.loaders
    x.prepare()
    x.train()
    x.save()
    x.eval()


if __name__ == "__main__":
    main()

"""
python glue.py \
  --model_name bert-base-cased \
  --task_name $TASK_NAME \
  --max_len 128 \
  --train_batch_size 32 \
  --lr 2e-5 \
  --out_dir /tmp/$TASK_NAME/

accelerate launch glue.py \
  --model_name bert-base-cased \
  --task_name $TASK_NAME \
  --max_len 128 \
  --train_batch_size 32 \
  --lr 2e-5 \
  --train_epochs 3 \
  --out_dir /tmp/$TASK_NAME/
"""
