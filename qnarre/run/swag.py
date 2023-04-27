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
# fine-tune for multiple choice

import logging
import random
import torch

from dataclasses import dataclass
from datasets import load_metric
from itertools import chain
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForChoicepleChoice, PreTrainedTokenizerBase

from .params import TRAIN, EVAL, ALL, EACH
from .runner import Runner as Base

log = logging.getLogger(__name__)


@dataclass
class DataCollatorForChoicepleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_len = None
    pad_to_multiple_of = None

    def __call__(self, xs):
        label_name = "label" if "label" in xs[0].keys() else "labels"
        labels = [x.pop(label_name) for x in xs]
        size = len(xs)
        choices = len(xs[0]["input_ids"])
        ys = [[{k: v[i] for k, v in x.items()} for i in range(choices)] for x in xs]
        ys = list(chain(*ys))
        ys = self.tokenizer.pad(
            ys,
            padding=self.padding,
            max_len=self.max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        ys = {k: v.view(size, choices, -1) for k, v in ys.items()}
        ys["labels"] = torch.tensor(labels, dtype=torch.int64)
        return ys


class Runner(Base):
    AutoModel = AutoModelForChoicepleChoice

    @property
    def cols(self):
        if self._cols is None:
            ds = self.dataset
            if ds[TRAIN] is not None:
                cs = ds[TRAIN].column_names
            else:
                cs = ds[EVAL].column_names
            e = [f"ending{x}" for x in range(4)]
            c = "sent1"
            q = "sent2"
            l = "label" if "label" in cs else "labels"
            self._cols = {ALL: cs, EACH: [e, c, q, l]}
        return self._cols

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
        e_col, c_col, q_col, l_col = self.cols[EACH]
        firsts = [[x] * 4 for x in xs[c_col]]
        qs = xs[q_col]
        seconds = [[f"{q} {xs[x][i]}" for x in e_col] for i, q in enumerate(qs)]
        firsts = list(chain(*firsts))
        seconds = list(chain(*seconds))
        ys = self.tokenizer(
            firsts,
            seconds,
            max_len=ps.max_len,
            padding=self.padding,
            truncation=True,
        )
        ys = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in ys.items()}
        ys["labels"] = xs[l_col]
        return ys

    @property
    def loaders(self):
        if self._loaders is None:
            ps, mgr = self.params, self.mgr
            if ps.pad_to_max_length:
                c = default_data_collator
            else:
                c = DataCollatorForChoicepleChoice(
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
            self._metric = load_metric("accuracy")
        return self._metric

    def eval_epoch(self, e):
        m, mgr = self.model, self.mgr
        m.eval()
        for xs in self.loaders[EVAL]:
            with torch.no_grad():
                ys = m(**xs)
            ys = ys.logits.argmax(dim=-1)
            self.metric.add_batch(predictions=mgr.gather(ys), references=mgr.gather(xs["labels"]))
        y = self.metric.compute()
        mgr.print(f"epoch {e}: {y}")


def main():
    x = Runner()
    x.dataset
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
accelerate launch swag.py \
--model_name bert-base-uncased \
--dataset_name swag \
--out_dir /tmp/test-swag-no-trainer \
--pad_to_max_length
"""
