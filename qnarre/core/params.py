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

import argparse

from transformers import MODEL_MAPPING, SchedulerType

TRAIN = "train"
EVAL = "validation"
TEST = "test"

ALL = "all"
EACH = "each"
LABEL = "label"

MODEL_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(c.model_type for c in MODEL_CLASSES)

LR_TYPES = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
]


question_answering_column_name_mapping = {
    "squad_v2": ("question", "context", "answer"),
}


def parse_params(xs):
    x = argparse.ArgumentParser()
    x.add_argument("--answer_column", type=str, default="answers")
    x.add_argument("--audio_column", type=str, default="audio")
    x.add_argument("--block_size", type=int, default=None)
    x.add_argument("--cache_dir", type=str, default=None)
    x.add_argument("--config_name", type=str, default=None)
    x.add_argument("--config_overrides", type=str, default=None)
    x.add_argument("--context_column", type=str, default="context")
    x.add_argument("--cuda", action="store_true")
    x.add_argument("--dataset_config", type=str, default=None)
    x.add_argument("--dataset_name", type=str, default=None)
    x.add_argument("--debug", action="store_true")
    x.add_argument("--do_eval", action="store_true")
    x.add_argument("--do_test", action="store_true")
    x.add_argument("--do_train", action="store_true")
    x.add_argument("--doc_stride", type=int, default=128)
    x.add_argument("--eval_batch_size", type=int, default=8)
    x.add_argument("--eval_file", type=str, default=None)
    x.add_argument("--feature_extractor", type=str, default=None)
    x.add_argument("--grad_accumulation_steps", type=int, default=1)
    x.add_argument("--hub_model_id", type=str)
    x.add_argument("--hub_token", type=str)
    x.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    x.add_argument("--label_all_tokens", action="store_true")
    x.add_argument("--label_column", type=str, default="label")
    x.add_argument("--language", type=str, default=None)
    x.add_argument("--line_by_line", type=bool, default=False)
    x.add_argument("--lower_case", type=bool, default=False)
    x.add_argument("--lr_scheduler", type=SchedulerType, default="linear", choices=LR_TYPES)
    x.add_argument("--lr", type=float, default=5e-5)
    x.add_argument("--max_answer_length", type=int, default=30)
    x.add_argument("--max_duration", type=float, default=20.0)
    x.add_argument("--max_eval_samples", type=int, default=None)
    x.add_argument("--max_len", type=int, default=128)
    x.add_argument("--max_seq_length", type=int, default=384)  # 512
    x.add_argument("--max_source_length", type=int, default=1024)
    x.add_argument("--max_span_length", type=int, default=5)
    x.add_argument("--max_target_length", type=int, default=128)
    x.add_argument("--max_test_samples", type=int, default=None)
    x.add_argument("--max_train_samples", type=int, default=None)
    x.add_argument("--max_train_steps", type=int, default=None)
    x.add_argument("--min_duration", type=float, default=0.0)
    x.add_argument("--mlm_probability", type=float, default=0.15)
    x.add_argument("--model_name", type=str, required=True)
    x.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    x.add_argument("--model_version", type=str, default="main")
    x.add_argument("--n_best_size", type=int, default=20)
    x.add_argument("--no_keep_linebreaks", action="store_true")
    x.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    x.add_argument("--n_beams", type=int, default=None)
    x.add_argument("--num_warmup_steps", type=int, default=0)
    x.add_argument("--num_workers", type=int, default=4)
    x.add_argument("--out_dir", type=str, default=None)
    x.add_argument("--overwrite_cache", type=bool, default=False)
    x.add_argument("--pad_to_max_length", action="store_true")
    x.add_argument("--plm_probability", type=float, default=1 / 6)
    x.add_argument("--push_to_hub", action="store_true")
    x.add_argument("--question_column", type=str, default="question")
    x.add_argument("--return_entity_metrics", action="store_true")
    x.add_argument("--seed", type=int, default=55)
    x.add_argument("--source_lang", type=str, default=None)
    x.add_argument("--source_prefix", type=str, default=None)
    x.add_argument("--split_percent", default=5)
    x.add_argument("--summary_column", type=str, default=None)
    x.add_argument("--target_lang", type=str, default=None)
    x.add_argument("--test_file", type=str, default=None)
    x.add_argument("--test_with_gen", type=bool, default=True)
    x.add_argument("--text_column", type=str, default="text")
    x.add_argument("--tokenizer_name", type=str, default=None)
    x.add_argument("--train_batch_size", type=int, default=8)
    x.add_argument("--train_epochs", type=int, default=3)
    x.add_argument("--train_file", type=str, default=None)
    x.add_argument("--train_language", type=str, default=None)
    x.add_argument("--use_auth_token", type=bool, default=False)
    x.add_argument("--use_fast_tokenizer", type=bool, default=True)
    x.add_argument("--use_slow_tokenizer", action="store_true")
    x.add_argument("--val_max_target_length", type=int, default=None)
    x.add_argument("--version_2_with_negative", type=bool, default=False)
    x.add_argument("--weight_decay", type=float, default=0.0)
    for n, kw in xs:
        x.add_argument(n, **kw)

    y = x.parse_args()

    if (
        y.dataset_name is None
        and y.train_file is None
        and y.eval_file is None
        and y.test_file is None
    ):
        raise ValueError("Need either a dataset name or a train/eval/test file")
    else:
        if y.train_file is not None:
            y = y.train_file.split(".")[-1]
            assert y in ["csv", "json", "txt"], "`train_file` should be a csv or a json file"
        if y.eval_file is not None:
            y = y.eval_file.split(".")[-1]
            assert y in ["csv", "json", "txt"], "`eval_file` should be a csv or a json file"
        if y.test_file is not None:
            y = y.test_file.split(".")[-1]
            assert y in ["csv", "json", "txt"], "`test_file` should be a csv or a json file"

    if y.push_to_hub:
        assert y.output_dir is not None, "Need an `output_dir` for repo with `--push_to_hub`"
    return y
