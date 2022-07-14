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
# fine-tune for masked language modeling (BERT, ALBERT, RoBERTa...)

import logging
import math
import random
import torch

from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

from .params import TRAIN, EVAL, ALL, EACH
from .runner import Runner as Base
from .utils import group_texts

log = logging.getLogger(__name__)


class Runner(Base):
    AutoModel = AutoModelForMaskedLM

    @property
    def dataset(self):
        if self._dataset is None:
            ps = self.params
            if ps.dataset_name is not None:
                y = load_dataset(ps.dataset_name, ps.dataset_config)
                if EVAL not in y.keys():
                    y[EVAL] = load_dataset(
                        ps.dataset_name, ps.dataset_config, split=f"train[:{ps.split_percent}%]"
                    )
                    y[TRAIN] = load_dataset(
                        ps.dataset_name, ps.dataset_config, split=f"train[{ps.split_percent}%:]"
                    )
            else:
                x, xs = None, {}
                if ps.eval_file is not None:
                    xs[EVAL] = x = ps.eval_file
                if ps.train_file is not None:
                    xs[TRAIN] = x = ps.train_file
                x = x.split(".")[-1]
                if x == "txt":
                    x = "text"
                y = load_dataset(x, data_files=xs)
                if EVAL not in y.keys():
                    y[EVAL] = load_dataset(x, data_files=xs, split=f"train[:{ps.split_percent}%]")
                    y[TRAIN] = load_dataset(x, data_files=xs, split=f"train[{ps.split_percent}%:]")
            self._dataset = y
        return self._dataset

    @property
    def cols(self):
        if self._cols is None:
            cs = self.dataset[TRAIN].column_names
            t = "text" if "text" in cs else cs[0]
            self._cols = {ALL: cs, EACH: [t]}
        return self._cols

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            ps, t = self.params, super().tokenizer
            if ps.max_seq_length is None:
                b = t.model_max_length
                if b > 1024:
                    log.warning(f"Using max_seq_length=1024")
                    b = 1024
            else:
                if ps.max_seq_length > t.model_max_length:
                    log.warning(f"Using max_seq_length={t.model_max_length}")
                b = min(ps.max_seq_length, t.model_max_length)
            self.max_seq_length = b
        return self._tokenizer

    @property
    def train_ds(self):
        if self._train_ds is None:
            ps, mgr, ds = self.params, self.mgr, self.dataset
            if ps.line_by_line:
                with mgr.main_process_first():
                    self._dataset = y = ds.map(
                        self.prep_for_train,
                        batched=True,
                        num_proc=ps.num_workers,
                        remove_columns=[self.cols[EACH][0]],
                        load_from_cache_file=not ps.overwrite_cache,
                        desc="Running tokenizer line_by_line",
                    )
            else:
                with mgr.main_process_first():
                    y = ds.map(
                        self.prep_for_train,
                        batched=True,
                        num_proc=ps.num_workers,
                        remove_columns=self.cols[ALL],
                        load_from_cache_file=not ps.overwrite_cache,
                        desc="Running tokenizer on every text",
                    )
                with mgr.main_process_first():
                    self._dataset = y = y.map(
                        partial(group_texts, self.max_seq_length),
                        batched=True,
                        num_proc=ps.num_workers,
                        load_from_cache_file=not ps.overwrite_cache,
                        desc=f"Grouping texts in blocks of {self.max_seq_length}",
                    )
            y = y[TRAIN]
            if ps.max_train_samples is not None:
                y = y.select(range(ps.max_train_samples))
            for i in random.sample(range(len(y)), 3):
                log.info(f"Sample {i} of the training set: {y[i]}")
            self._train_ds = y
        return self._train_ds

    def prep_for_train(self, xs):
        ps, c = self.params, self.cols[EACH][0]
        if ps.line_by_line:
            xs[c] = [x for x in xs[c] if len(x) > 0 and not x.isspace()]
            return self.tokenizer(
                xs[c],
                padding=self.padding,
                truncation=True,
                max_len=self.max_seq_length,
                return_special_tokens_mask=True,
            )
        else:
            return self.tokenizer(xs[c], return_special_tokens_mask=True)

    @property
    def loaders(self):
        if self._loaders is None:
            ps = self.params
            c = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=ps.mlm_probability)
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
        return self._loaders

    def eval_epoch(self, e):
        m, mgr = self.model, self.mgr
        m.eval()
        y = []
        for xs in self.loaders[EVAL]:
            with torch.no_grad():
                ys = m(**xs)
            y.append(mgr.gather(ys.loss.repeat(self.params.eval_batch_size)))
        y = torch.cat(y)[: len(self.eval_ds)]
        try:
            y = math.exp(torch.mean(y))
        except OverflowError:
            y = float("inf")
        mgr.print(f"epoch {e}: perplexity: {y}")


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
