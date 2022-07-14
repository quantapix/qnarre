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
# fine-tune for token classification (NER, POS, CHUNKS)

import logging
import random
import torch

from datasets import ClassLabel, load_metric
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    default_data_collator,
)

from .params import TRAIN, EVAL, TEST, ALL, EACH
from .runner import Runner as Base
from .utils import get_list

log = logging.getLogger(__name__)


class Runner(Base):
    AutoModel = AutoModelForTokenClassification

    @property
    def cols(self):
        if self._cols is None:
            ps, ds = self.params, self.dataset
            x = ds[TRAIN] or ds[EVAL] or ds[TEST]
            cs = x.column_names
            fs = x.features
            if ps.text_column is not None:
                t = ps.text_column
            else:
                t = "tokens" if "tokens" in cs else cs[0]
            if ps.label_column is not None:
                l = ps.label_column
            else:
                l = f"{ps.task_name}_tags" if f"{ps.task_name}_tags" in cs else cs[1]
            if isinstance(fs[l].feature, ClassLabel):
                self.labels = ls = fs[l].feature.names
                self.ids = {i: i for i in range(len(ls))}
            else:
                self.labels = ls = get_list(ds[TRAIN][l])
                self.ids = {l: i for i, l in enumerate(ls)}
            self.b_to_i = []
            for i, x in enumerate(ls):
                if x.startswith("B-") and x.replace("B-", "I-") in ls:
                    self.b_to_i.append(ls.index(x.replace("B-", "I-")))
                else:
                    self.b_to_i.append(i)
            self._cols = {ALL: cs, EACH: [t, l]}
        return self._cols

    @property
    def config(self):
        if self._config is None:
            ps = self.params
            x = ps.config_name if ps.config_name else ps.model_name
            if x:
                y = AutoConfig.from_pretrained(x, n_labels=len(self.labels))
            else:
                y = CONFIG_MAPPING[ps.model_type]()
                log.warning("Creating new config")
            self._config = y
        return self._config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            ps = self.params
            x = ps.tokenizer_name if ps.tokenizer_name else ps.model_name
            if not x:
                raise ValueError("Tokenizer from scratch is not supported")
            if self.config.model_type in {"gpt2", "roberta"}:
                y = AutoTokenizer.from_pretrained(x, use_fast=True, add_prefix_space=True)
            else:
                y = AutoTokenizer.from_pretrained(x, use_fast=True)
            self._tokenizer = y
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            ls = self.labels
            m = super().model
            if m.config.label2id != PretrainedConfig(n_labels=len(self.labels)).label2id:
                ids = {l: i for l, i in m.config.label2id.items()}
                if list(sorted(ids.keys())) == list(sorted(ls)):
                    log.info(f"Using config label map: {ids}")
                    self.ids = {l(ids[l]) for l in ls}
                else:
                    log.warning(
                        f"Ignoring mismatched {list(sorted(ids.keys()))} vs {list(sorted(ls))}"
                    )
            else:
                self.ids = {l: i for i, l in enumerate(ls)}
            m.config.label2id = self.ids
            m.config.id2label = {i: l for l, i in self.ids.items()}
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
        ps = self.params
        t, l = self.cols[EACH]
        ys = self.tokenizer(
            xs[t],
            max_len=ps.max_len,
            padding=self.padding,
            truncation=True,
            is_split_into_words=True,
        )
        labels = []
        for i, x in enumerate(xs[l]):
            ids = []
            prev = None
            ws = ys.word_ids(batch_index=i)
            for w in ws:
                if w is None:
                    ids.append(-100)
                elif w != prev:
                    ids.append(self.ids[x[w]])
                else:
                    if ps.label_all_tokens:
                        ids.append(self.b_to_i[self.ids[x[w]]])
                    else:
                        ids.append(-100)
                prev = w
            labels.append(ids)
        ys["labels"] = labels
        return ys

    @property
    def loaders(self):
        if self._loaders is None:
            ps, mgr = self.params, self.mgr
            if ps.pad_to_max_length:
                c = default_data_collator
            else:
                c = DataCollatorForTokenClassification(
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
            self._metric = load_metric("seqeval")
        return self._metric

    def eval_epoch(self, e):
        m, mgr = self.model, self.mgr
        m.eval()
        for xs in self.loaders[EVAL]:
            with torch.no_grad():
                ys = m(**xs)
            ys = ys.logits.argmax(dim=-1)
            ls = xs["labels"]
            if not self.params.pad_to_max_length:
                ys = mgr.pad_across_processes(ys, dim=1, PAD=-100)
                ls = mgr.pad_across_processes(ls, dim=1, PAD=-100)
            ys, ls = self.get_labels(mgr.gather(ys), mgr.gather(ls))
            self.metric.add_batch(predictions=ys, references=ls)
        y = self.calc_metrics()
        mgr.print(f"epoch {e}: {y}")

    def get_labels(self, xs, ls):
        mgr = self.mgr
        if mgr.device.type == "cpu":
            xs = xs.detach().clone().numpy()
            ls = ls.detach().clone().numpy()
        else:
            xs = xs.detach().cpu().clone().numpy()
            ls = ls.detach().cpu().clone().numpy()
        ys = [[self.labels[x2] for (x2, l2) in zip(x, l) if l2 != -100] for x, l in zip(xs, ls)]
        yl = [[self.labels[l2] for (_, l2) in zip(x, l) if l2 != -100] for x, l in zip(xs, ls)]
        return ys, yl

    def calc_metrics(self):
        ps, xs = self.params, self.metric.compute()
        if ps.return_entity_metrics:
            ys = {}
            for k, v in xs.items():
                if isinstance(v, dict):
                    for n, v in v.items():
                        ys[f"{k}_{n}"] = v
                else:
                    ys[k] = v
            return ys
        else:
            return {
                "precision": xs["overall_precision"],
                "recall": xs["overall_recall"],
                "f1": xs["overall_f1"],
                "accuracy": xs["overall_accuracy"],
            }


def main():
    ps = [("--task_name", {"type", "default": "ner", "choices": ["ner", "pos", "chunk"]})]
    x = Runner(ps)
    x.dataset
    x.cols
    x.config
    x.tokenizer
    x.model
    x.model.resize_token_embeddings(len(x.tokenizer))
    x.loaders
    x.prepare()
    x.train()
    x.save()


if __name__ == "__main__":
    main()

"""
python ner.py \
  --model_name bert-base-cased \
  --dataset_name conll2003 \
  --task_name ner \
  --train_batch_size 32 \
  --lr 2e-5 \
  --out_dir /tmp/ner

accelerate config
accelerate test

accelerate launch ner.py \
  --model_name bert-base-uncased \
  --dataset_name conll2003 \
  --task_name ner \
  --train_batch_size 32 \
  --lr 2e-5 \
  --out_dir /tmp/ner
  --pad_to_max_length \
  --return_entity_metrics
"""
