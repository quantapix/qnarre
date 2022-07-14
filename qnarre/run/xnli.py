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
# fine-tune multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM)

import logging
import random

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    default_data_collator,
)

from .params import TRAIN, EVAL, TEST, ALL, EACH
from .runner import Runner as Base

log = logging.getLogger(__name__)


class Runner(Base):
    AutoModel = AutoModelForSequenceClassification

    @property
    def dataset(self):
        if self._dataset is None:
            ps = self.params
            y = {TRAIN: {}, EVAL: {}, TEST: {}}
            if ps.do_train:
                if ps.train_language is None:
                    y[TRAIN] = load_dataset(
                        "xnli", ps.language, split=TRAIN, cache_dir=ps.cache_dir
                    )
                else:
                    y[TRAIN] = load_dataset(
                        "xnli", ps.train_language, split=TRAIN, cache_dir=ps.cache_dir
                    )
                self.label_list = y.features["label"].names
            if ps.do_eval:
                y[EVAL] = load_dataset("xnli", ps.language, split=EVAL, cache_dir=ps.cache_dir)
                self.label_list = y.features["label"].names
            if ps.do_test:
                y[TEST] = load_dataset("xnli", ps.language, split=TEST, cache_dir=ps.cache_dir)
                self.label_list = y.features["label"].names
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
    def config(self):
        if self._config is None:
            ps = self.params
            x = ps.config_name if ps.config_name else ps.model_name
            if x:
                y = self.AutoConfig.from_pretrained(
                    x,
                    n_labels=len(self.label_list),
                    finetune="xnli",
                    cache_dir=ps.cache_dir,
                    revision=ps.model_version,
                    use_auth_token=True if ps.use_auth_token else None,
                )
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
            y = self.AutoTokenizer.from_pretrained(
                x,
                lower_case=ps.lower_case,
                cache_dir=ps.cache_dir,
                use_fast=ps.use_fast_tokenizer,
                revision=ps.model_version,
                use_auth_token=True if ps.use_auth_token else None,
            )
            self._tokenizer = y
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
                    load_from_cache_file=not ps.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            for i in random.sample(range(len(y)), 3):
                log.info(f"Sample {i} of the training set: {y[i]}")
            self._train_ds = y
        return self._train_ds

    def prep_for_train(self, xs):
        return self.tokenizer(
            xs["premise"],
            xs["hypothesis"],
            padding=self.padding,
            max_len=self.params.max_seq_length,
            truncation=True,
        )

    @property
    def eval_ds(self):
        if self._eval_ds is None:
            ps, mgr = self.params, self.mgr
            y = super().eval_ds
            with mgr.main_process_first():
                y = y.map(
                    self.prep_for_train,
                    batched=True,
                    load_from_cache_file=not ps.overwrite_cache,
                    desc="Running tokenizer on eval dataset",
                )
            self._eval_ds = y
        return self._eval_ds

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
    def loaders(self):
        if self._loaders is None:
            ps = self.params
            if ps.pad_to_max_length:
                c = default_data_collator
            elif ps.fp16:
                c = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
            else:
                c = None
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
            if ps.do_test:
                p = DataLoader(self.test_ds, collate_fn=c, batch_size=ps.eval_batch_size)
                self._loaders[TEST] = p
        return self._loaders

    @property
    def metric(self):
        if self._metric is None:
            self._metric = load_metric("xnli")
            # def compute_metrics(p: EvalPrediction):
            #    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            #    preds = np.argmax(preds, axis=1)
            #    return metric.compute(predictions=preds, references=p.label_ids)
        return self._metric


def main():
    x = Runner()
    x.dataset
    x.config
    x.tokenizer
    x.model
    # x.model.resize_token_embeddings(len(x.tokenizer))
    x.loaders
    x.prepare()
    x.train()
    x.save()


if __name__ == "__main__":
    main()

"""
python xnli.py \
  --model_name bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --train_batch_size 32 \
  --train_epochs 2.0 \
  --max_seq_length 128 \
  --out_dir /tmp/debug_xnli/ \
  --save_steps -1
"""
