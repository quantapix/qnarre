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
# fine-tune for permutation language modeling

import logging

from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForPermutationLanguageModeling,
    XLNetConfig,
    XLNetLMHeadModel,
)

from .mlm import Runner as Mlm

from .params import TRAIN, EVAL, ALL, EACH
from .runner import Runner as Base
from .utils import group_texts

log = logging.getLogger(__name__)


class Runner(Base):
    AutoModel = XLNetLMHeadModel

    @property
    def dataset(self):
        if self._dataset is None:
            ps = self.params
            if ps.dataset_name is not None:
                y = load_dataset(ps.dataset_name, ps.dataset_config, cache_dir=ps.cache_dir)
                if EVAL not in y.keys():
                    y[EVAL] = load_dataset(
                        ps.dataset_name,
                        ps.dataset_config,
                        split=f"train[:{ps.split_percent}%]",
                        cache_dir=ps.cache_dir,
                    )
                    y[TRAIN] = load_dataset(
                        ps.dataset_name,
                        ps.dataset_config,
                        split=f"train[{ps.split_percent}%:]",
                        cache_dir=ps.cache_dir,
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
                y = load_dataset(x, data_files=xs, cache_dir=ps.cache_dir)
                if EVAL not in y.keys():
                    y[EVAL] = load_dataset(
                        x,
                        data_files=xs,
                        split=f"train[:{ps.split_percent}%]",
                        cache_dir=ps.cache_dir,
                    )
                    y[TRAIN] = load_dataset(
                        x,
                        data_files=xs,
                        split=f"train[{ps.split_percent}%:]",
                        cache_dir=ps.cache_dir,
                    )
            self._dataset = y
        return self._dataset

    @property
    def cols(self):
        if self._cols is None:
            ps = self.params
            cs = self.dataset[TRAIN if ps.do_train else EVAL].column_names
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
                    cache_dir=ps.cache_dir,
                    revision=ps.model_version,
                    use_auth_token=True if ps.use_auth_token else None,
                )
            else:
                y = XLNetConfig()
                log.warning("Creating new config")
                if ps.config_overrides is not None:
                    log.info(f"Overriding config: {ps.config_overrides}")
                    y.update_from_string(ps.config_overrides)
                    log.info(f"New config: {y}")
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
                use_fast=ps.use_fast_tokenizer,
                revision=ps.model_version,
                use_auth_token=True if ps.use_auth_token else None,
            )
            self._tokenizer = y
            if ps.max_seq_length is None:
                b = y.model_max_length
                if b > 1024:
                    log.warning(f"Using max_seq_length=1024")
                    b = 1024
            else:
                if ps.max_seq_length > y.model_max_length:
                    log.warning(f"Using max_seq_length={y.model_max_length}")
                b = min(ps.max_seq_length, y.model_max_length)
            self.max_seq_length = b
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

    train_ds = Mlm.train_ds

    def prep_for_train(self, xs):
        ps, c = self.params, self.cols[EACH][0]
        if ps.line_by_line:
            xs[c] = [x for x in xs[c] if len(x) > 0 and not x.isspace()]
            return self.tokenizer(
                xs[c], padding=self.padding, truncation=True, max_len=self.max_seq_length
            )
        else:
            return self.tokenizer(xs[c])

    @property
    def loaders(self):
        if self._loaders is None:
            ps = self.params
            c = DataCollatorForPermutationLanguageModeling(
                self.tokenizer,
                plm_probability=ps.plm_probability,
                max_span_length=ps.max_span_length,
            )
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
        return self._loaders

    eval_epoch = Mlm.eval_epoch


def main():
    ps = [("--max_seq_length", {"type", "default": 512})]
    x = Runner(ps)
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
