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
# fine-tune on text translation

import logging
import numpy as np
import random
import torch

from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    default_data_collator,
)

from .params import TRAIN, EVAL, ALL
from .runner import Runner as Base

log = logging.getLogger(__name__)


def postproc(xs, ls):
    xs = [x.strip() for x in xs]
    ls = [[x.strip()] for x in ls]
    return xs, ls


class Runner(Base):
    AutoModel = AutoModelForSeq2SeqLM

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            ps = self.params
            t = super().tokenizer
            if isinstance(t, (MBartTokenizer, MBartTokenizerFast)):
                if ps.source_lang is not None:
                    t.src_lang = ps.source_lang
                if ps.target_lang is not None:
                    t.tgt_lang = ps.target_lang
            self.source_lang = ps.source_lang.split("_")[0]
            self.target_lang = ps.target_lang.split("_")[0]
            self.prefix = ps.source_prefix if ps.source_prefix is not None else ""
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            ps = self.params
            t, m = self.tokenizer, super().model
            if m.config.dec_START is None and isinstance(t, (MBartTokenizer, MBartTokenizerFast)):
                assert (
                    ps.target_lang is not None and ps.source_lang is not None
                ), "mBart needs --target_lang and --source_lang"
                if isinstance(t, MBartTokenizer):
                    m.config.dec_START = t.lang_code_to_id[ps.target_lang]
                else:
                    m.config.dec_START = t.convert_tokens_to_ids(ps.target_lang)
            if m.config.dec_START is None:
                raise ValueError("Needs `config.dec_START`")

    @property
    def train_ds(self):
        if self._train_ds is None:
            ps, mgr, ds = self.params, self.mgr, self.dataset
            with mgr.main_process_first():
                self._dataset = y = ds.map(
                    self.prep_for_train,
                    batched=True,
                    remove_columns=self.cols[ALL],
                    load_from_cache_file=not ps.overwrite_cache,
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
        ps, t = self.params, self.tokenizer
        ins = [x[self.source_lang] for x in xs["translation"]]
        targets = [x[self.target_lang] for x in xs["translation"]]
        ins = [self.prefix + x for x in ins]
        ys = t(ins, max_len=ps.max_source_length, padding=ps.padding, truncation=True)
        with t.as_target_tokenizer():
            ls = t(targets, max_len=ps.max_target_length, padding=ps.padding, truncation=True)
        if self.padding == "max_len" and ps.ignore_pad_token_for_loss:
            ls["input_ids"] = [[(y if y != t.PAD else -100) for y in x] for x in ls["input_ids"]]
        ys["labels"] = ls["input_ids"]
        return ys

    @property
    def loaders(self):
        if self._loaders is None:
            ps, t = self.params, self.tokenizer
            if ps.pad_to_max_length:
                c = default_data_collator
            else:
                c = DataCollatorForSeq2Seq(
                    t,
                    model=self.model,
                    label_pad_token_id=-100 if ps.ignore_pad_token_for_loss else t.PAD,
                    pad_to_multiple_of=8 if self.mgr.use_fp16 else None,
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
            self._metric = load_metric("sacrebleu")
        return self._metric

    def eval_epoch(self, e):
        ps, t, m, mgr = self.params, self.tokenizer, self.model, self.mgr
        m.eval()
        if ps.val_max_target_length is None:
            ps.val_max_target_length = ps.max_target_length
        kw = {
            "max_len": ps.val_max_target_length if ps is not None else self.config.max_len,
            "n_beams": ps.n_beams,
        }
        for xs in self.loaders[EVAL]:
            with torch.no_grad():
                ys = mgr.unwrap_model(m).generate(xs["input_ids"], mask=xs["mask"], **kw)
                ys = mgr.pad_across_processes(ys, dim=1, PAD=t.PAD)
                ls = xs["labels"]
                if not ps.pad_to_max_length:
                    ls = mgr.pad_across_processes(xs["labels"], dim=1, PAD=t.PAD)
                ys = mgr.gather(ys).cpu().numpy()
                ls = mgr.gather(ls).cpu().numpy()
                if ps.ignore_pad_token_for_loss:
                    ls = np.where(ls != -100, ls, t.PAD)
                ys = t.batch_decode(ys, skip_special_tokens=True)
                ls = t.batch_decode(ls, skip_special_tokens=True)
                ys, ls = postproc(ys, ls)
                self.metric.add_batch(predictions=ys, references=ls)
        y = self.metric.compute()["score"]
        mgr.print(f"epoch {e}: bleu: {y}")


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
python xlate.py \
    --model_name Helsinki-NLP/opus-mt-en-ro \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config ro-en \
    --out_dir ~/tmp/tst-translation

accelerate launch xlate.py \
    --model_name Helsinki-NLP/opus-mt-en-ro \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config ro-en \
    --out_dir ~/tmp/tst-translation
"""
