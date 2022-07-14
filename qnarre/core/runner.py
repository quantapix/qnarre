# Copyright 2022 Quantapix Authors. All Rights Reserved.
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

import datasets
import logging
import math
import os
import transformers

from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import Repository
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.file_utils import get_full_repo_name
from transformers import (
    CONFIG_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from .params import parse_params, TRAIN, EVAL, TEST, ALL


log = logging.getLogger(__name__)


class Runner:
    AutoModel = AutoModel
    AutoConfig = AutoConfig
    AutoTokenizer = AutoTokenizer

    def __init__(self, xs=[]):
        self.params = ps = parse_params(xs)
        self.mgr = mgr = Accelerator()
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        log.info(mgr.state)
        log.setLevel(logging.INFO if mgr.is_local_main_process else logging.ERROR)
        if mgr.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        if ps.seed is not None:
            set_seed(ps.seed)
        if mgr.is_main_process:
            if ps.push_to_hub:
                if ps.hub_model_id is None:
                    x = get_full_repo_name(Path(ps.out_dir).name, token=ps.hub_token)
                else:
                    x = ps.hub_model_id
                self.repo = Repository(ps.out_dir, clone_from=x)
            elif ps.out_dir is not None:
                os.makedirs(ps.out_dir, exist_ok=True)
        mgr.wait_for_everyone()
        self.padding = "max_len" if ps.pad_to_max_length else False

    @property
    def dataset(self):
        if self._dataset is None:
            ps = self.params
            if ps.dataset_name is not None:
                y = load_dataset(ps.dataset_name, ps.dataset_config)
            else:
                x, xs = None, {}
                if ps.test_file is not None:
                    xs[TEST] = x = ps.test_file
                if ps.eval_file is not None:
                    xs[EVAL] = x = ps.eval_file
                if ps.train_file is not None:
                    xs[TRAIN] = x = ps.train_file
                y = load_dataset(x.split(".")[-1], data_files=xs)  # field="data")
            if ps.debug:
                for k in y.keys():
                    y[k] = y[k].select(range(100))
            self._dataset = y
        return self._dataset

    @property
    def cols(self):
        if self._cols is None:
            cs = self.dataset[TRAIN].column_names
            self._cols = {ALL: cs}
        return self._cols

    @property
    def config(self):
        if self._config is None:
            ps = self.params
            x = ps.config_name if ps.config_name else ps.model_name
            if x:
                y = self.AutoConfig.from_pretrained(x)
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
            y = self.AutoTokenizer.from_pretrained(x, use_fast=not ps.use_slow_tokenizer)
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
                )
            else:
                log.info("Training new model")
                y = self.AutoModel.from_config(self.config)
            self._model = y
        return self._model

    @property
    def eval_ds(self):
        if self._eval_ds is None:
            ps, ds = self.params, self.dataset
            y = ds[EVAL]
            if ps.max_eval_samples is not None:
                y = y.select(range(ps.max_eval_samples))
            self._eval_ds = y
        return self._eval_ds

    @property
    def test_ds(self):
        if self._test_ds is None:
            ps, ds = self.params, self.dataset
            y = ds[TEST]
            if ps.max_test_samples is not None:
                y = y.select(range(ps.max_test_samples))
            self._test_ds = y
        return self._test_ds

    @property
    def loaders(self):
        if self._loaders is None:
            ps = self.params
            c = default_data_collator
            t = DataLoader(
                self.train_ds, shuffle=True, collate_fn=c, batch_size=ps.train_batch_size
            )
            e = DataLoader(self.eval_ds, collate_fn=c, batch_size=ps.eval_batch_size)
            self._loaders = {TRAIN: t, EVAL: e}
        return self._loaders

    @property
    def optimizer(self):
        if self._optimizer is None:
            ps, m = self.params, self.model
            ds = ["bias", "LayerNorm.weight"]
            xs = [
                {
                    "params": [p for n, p in m.named_parameters() if not any(d in n for d in ds)],
                    "weight_decay": ps.weight_decay,
                },
                {
                    "params": [p for n, p in m.named_parameters() if any(d in n for d in ds)],
                    "weight_decay": 0.0,
                },
            ]
            self._optimizer = AdamW(xs, lr=ps.lr)
        return self._optimizer

    @property
    def scheduler(self):
        if self._scheduler is None:
            ps = self.params
            self._scheduler = get_scheduler(
                name=ps.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=ps.num_warmup_steps,
                num_training_steps=ps.max_train_steps,
            )
        return self._optimizer

    def prepare(self):
        m, mgr, ls = self.model, self.mgr, self.loaders
        m.to(mgr.device)
        t, e = ls[TRAIN], ls[EVAL]
        self._model, self._optimizer, t, e = mgr.prepare(m, self.optimizer, t, e)
        self._loaders = {TRAIN: t, EVAL: e}

    def train(self):
        ps, mgr, src = self.params, self.mgr, self.loaders[TRAIN]
        x = math.ceil(len(src) / ps.grad_accumulation_steps)
        if ps.max_train_steps is None:
            ps.max_train_steps = ps.num_train_epochs * x
        else:
            ps.num_train_epochs = math.ceil(ps.max_train_steps / x)
        m, o, s = self.model, self.optimizer, self.scheduler
        b = ps.train_batch_size * mgr.num_processes * ps.grad_accumulation_steps
        log.info("*** Training ***")
        log.info(f"  Num samples = {len(self.train_ds)}")
        log.info(f"  Num epochs = {ps.num_train_epochs}")
        log.info(f"  Batch size per device = {ps.train_batch_size}")
        log.info(f"  Batch size (w. parallel, distributed & accumulation) = {b}")
        log.info(f"  Grad accumulation steps = {ps.grad_accumulation_steps}")
        log.info(f"  Train steps = {ps.max_train_steps}")
        n = 0
        bar = tqdm(range(ps.max_train_steps), disable=not mgr.is_local_main_process)
        for e in range(ps.num_train_epochs):
            m.train()
            for i, xs in enumerate(src):
                ys = m(**xs)
                mgr.backward(ys.loss / ps.grad_accumulation_steps)
                if i % ps.grad_accumulation_steps == 0 or i == len(src) - 1:
                    o.step()
                    s.step()
                    o.zero_grad()
                    bar.update(1)
                    n += 1
                if n >= ps.max_train_steps:
                    break
            self.eval_epoch(e)
            if ps.push_to_hub and e < ps.num_train_epochs - 1:
                mgr.wait_for_everyone()
                mgr.unwrap_model(m).save_pretrained(ps.out_dir, save_function=mgr.save)
                if mgr.is_main_process:
                    self.tokenizer.save_pretrained(ps.out_dir)
                    self.repo.push_to_hub(commit_message=f"Training... epoch {e}", blocking=False)

    def eval_epoch(self, _):
        pass

    def save(self):
        ps, mgr = self.params, self.mgr
        if ps.out_dir is not None:
            mgr.wait_for_everyone()
            mgr.unwrap_model(self.model).save_pretrained(ps.out_dir, save_function=mgr.save)
            if mgr.is_main_process:
                self.tokenizer.save_pretrained(ps.out_dir)
                if ps.push_to_hub:
                    self.repo.push_to_hub(commit_message="End of training")
