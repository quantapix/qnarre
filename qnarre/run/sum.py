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
# fine-tune on summarization

import logging
import nltk
import numpy as np
import random
import torch

from datasets import load_metric
from filelock import FileLock
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers.file_utils import is_offline_mode

from .params import TRAIN, EVAL, ALL, EACH
from .runner import Runner as Base


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

COLS = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

log = logging.getLogger(__name__)


def postproc(xs, ls):
    xs = [x.strip() for x in xs]
    ls = [x.strip() for x in ls]
    xs = ["\n".join(nltk.sent_tokenize(x)) for x in xs]
    ls = ["\n".join(nltk.sent_tokenize(x)) for x in ls]
    return xs, ls


class Runner(Base):
    AutoModel = AutoModelForSeq2SeqLM

    def __init__(self:
        super().__init__()
        ps = self.params
        if ps.source_prefix is None and ps.model_name in [
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
        ]:
            log.warning("Running a t5 model without source prefix")
        self.prefix = ps.source_prefix if ps.source_prefix is not None else ""

    @property
    def cols(self):
        if self._cols is None:
            ps, cs = self.params, self.dataset[TRAIN].column_names
            xs = COLS.get(ps.dataset_name, None)
            if ps.text_column is None:
                t = xs[0] if xs is not None else cs[0]
            else:
                t = ps.text_column
                if t not in cs:
                    raise ValueError(f"--text_column={ps.text_column}' should be: {', '.join(cs)}")
            if ps.summary_column is None:
                s = xs[1] if xs is not None else cs[1]
            else:
                s = ps.summary_column
                if s not in cs:
                    raise ValueError(
                        f"--summary_column={ps.summary_column}' should be: {', '.join(cs)}"
                    )
            self._cols = {ALL: cs, EACH: [t, s]}
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
        ps = self.params
        t_col, s_col = self.cols[EACH]
        ins, outs = xs[t_col], xs[s_col]
        ins = [self.prefix + x for x in ins]
        tok = self.tokenizer
        ys = tok(ins, max_len=ps.max_source_length, padding=self.padding, truncation=True)
        with tok.as_target_tokenizer():
            ls = tok(outs, max_len=ps.max_target_length, padding=self.padding, truncation=True)
        if self.padding == "max_len" and ps.ignore_pad_token_for_loss:
            ls["input_ids"] = [
                [(x if x != tok.PAD else -100) for x in l] for l in ls["input_ids"]
            ]
        ys["labels"] = ls["input_ids"]
        return ys

    @property
    def loaders(self):
        if self._loaders is None:
            ps, mgr = self.params, self.mgr
            c = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=-100
                if ps.ignore_pad_token_for_loss
                else self.tokenizer.PAD,
                pad_to_multiple_of=8 if mgr.use_fp16 else None,
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
            self._metric = load_metric("rouge")
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
                ys = mgr.unwrap_model(m).generate(
                    xs["input_ids"],
                    mask=xs["mask"],
                    **kw,
                )
                p = t.PAD
                ys = mgr.pad_across_processes(ys, dim=1, PAD=p)
                ls = xs["labels"]
                if not ps.pad_to_max_length:
                    ls = mgr.pad_across_processes(ls, dim=1, PAD=p)
                ys = mgr.gather(ys).cpu().numpy()
                ls = mgr.gather(ls).cpu().numpy()
                if ps.ignore_pad_token_for_loss:
                    ls = np.where(ls != -100, ls, p)
                if isinstance(ys, tuple):
                    ys = ys[0]
                ys = t.batch_decode(ys, skip_special_tokens=True)
                ls = t.batch_decode(ls, skip_special_tokens=True)
                ys, ls = postproc(ys, ls)
                self.metric.add_batch(predictions=ys, references=ls)
        y = self.metric.compute(use_stemmer=True)
        y = {k: v.mid.fmeasure * 100 for k, v in y.items()}
        y = {k: round(v, 4) for k, v in y.items()}
        mgr.print(f"epoch {e}: {y}")


def main():
    x = Runner()
    x.dataset
    x.config
    x.tokenizer
    x.model
    x.model.resize_token_embeddings(len(x.tokenizer))
    if x.model.config.dec_START is None:
        raise ValueError("Needs `config.dec_START`")
    x.loaders
    x.prepare()
    x.train()
    x.save()


if __name__ == "__main__":
    main()


"""
python sum.py \
    --model_name t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --out_dir ~/tmp/tst-summarization

accelerate launch sum.py \
    --model_name t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --out_dir ~/tmp/tst-summarization
"""
