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

import os
import re

import numpy as np
import tensorflow as tf

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from ...utils import logging


_REALM_BLOCK_RECORDS_FILENAME = "block_records.npy"


log = logging.get_logger(__name__)


def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    tf_path = abspath(tf_checkpoint_path)
    log.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        if isinstance(model, RealmReader) and "reader" not in name:
            log.info(f"Skipping {name} as it is not {model.__class__.__name__}'s parameter")
            continue

        # For pretrained openqa reader
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(
            model, RealmForOpenQA
        ):
            name = name.replace("bert/", "reader/realm/")
            name = name.replace("cls/", "reader/cls/")

        # For pretrained encoder
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(
            model, RealmKnowledgeAugEncoder
        ):
            name = name.replace("bert/", "realm/")

        # For finetuned reader
        if name.startswith("reader"):
            reader_prefix = "" if isinstance(model, RealmReader) else "reader/"
            name = name.replace("reader/module/bert/", f"{reader_prefix}realm/")
            name = name.replace("reader/module/cls/", f"{reader_prefix}cls/")
            name = name.replace("reader/dense/", f"{reader_prefix}qa_outputs/dense_intermediate/")
            name = name.replace("reader/dense_1/", f"{reader_prefix}qa_outputs/dense_output/")
            name = name.replace(
                "reader/layer_normalization", f"{reader_prefix}qa_outputs/layer_normalization"
            )

        # For embedder and scorer
        if name.startswith("module/module/module/"):  # finetuned
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace(
                "module/module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/"
            )
            name = name.replace("module/module/module/dense/", f"{embedder_prefix}cls/dense/")
            name = name.replace(
                "module/module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/"
            )
            name = name.replace("module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace(
                "module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/"
            )
        elif name.startswith("module/module/"):  # pretrained
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
            name = name.replace("module/module/dense/", f"{embedder_prefix}cls/dense/")

        name = name.split("/")
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            log.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    log.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        log.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def convert_tfrecord_to_np(block_records_path, num_block_records):
    import tensorflow.compat.v1 as tf

    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())

    return np_record


class ScaNNSearcher:
    def __init__(
        self,
        db,
        num_neighbors,
        dimensions_per_block=2,
        num_leaves=1000,
        num_leaves_to_search=100,
        training_sample_size=100000,
    ):
        from scann.scann_ops.py.scann_ops_pybind import builder as Builder

        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure="dot_product")
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
        )
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

        self.searcher = builder.build()

    def search_batched(self, question_projection):
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        return retrieved_block_ids.astype("int64")


class RealmRetriever:
    def __init__(self, block_records, tokenizer):
        super().__init__()
        self.block_records = block_records
        self.tokenizer = tokenizer

    def __call__(
        self,
        retrieved_block_ids,
        question_input_ids,
        answer_ids,
        max_length=None,
        return_tensors="pt",
    ):
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids, axis=0)

        question = self.tokenizer.decode(question_input_ids[0], skip_special_tokens=True)

        text = []
        text_pair = []
        for retrieved_block in retrieved_blocks:
            text.append(question)
            text_pair.append(retrieved_block.decode())

        concat_inputs = self.tokenizer(
            text,
            text_pair,
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
            max_length=max_length,
        )
        concat_inputs_tensors = concat_inputs.convert_to_tensors(return_tensors)

        if answer_ids is not None:
            return self.block_has_answer(concat_inputs, answer_ids) + (concat_inputs_tensors,)
        else:
            return (None, None, None, concat_inputs_tensors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kw,
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            block_records_path = os.path.join(
                pretrained_model_name_or_path, _REALM_BLOCK_RECORDS_FILENAME
            )
        else:
            block_records_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=_REALM_BLOCK_RECORDS_FILENAME,
                **kw,
            )
        block_records = np.load(block_records_path, allow_pickle=True)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kw)

        return cls(block_records, tokenizer)

    def save_pretrained(self, save_directory):
        # save block records
        np.save(os.path.join(save_directory, _REALM_BLOCK_RECORDS_FILENAME), self.block_records)
        # save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    def block_has_answer(self, concat_inputs, answer_ids):
        has_answers = []
        start_pos = []
        end_pos = []
        max_answers = 0

        for input_id in concat_inputs.input_ids:
            input_id_list = input_id.tolist()
            # Check answers between two [SEP] tokens
            first_sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
            second_sep_idx = (
                first_sep_idx
                + 1
                + input_id_list[first_sep_idx + 1 :].index(self.tokenizer.sep_token_id)
            )

            start_pos.append([])
            end_pos.append([])
            for answer in answer_ids:
                for idx in range(first_sep_idx + 1, second_sep_idx):
                    if answer[0] == input_id_list[idx]:
                        if input_id_list[idx : idx + len(answer)] == answer:
                            start_pos[-1].append(idx)
                            end_pos[-1].append(idx + len(answer) - 1)

            if len(start_pos[-1]) == 0:
                has_answers.append(False)
            else:
                has_answers.append(True)
                if len(start_pos[-1]) > max_answers:
                    max_answers = len(start_pos[-1])

        # Pad -1 to max_answers
        for start_pos_, end_pos_ in zip(start_pos, end_pos):
            if len(start_pos_) < max_answers:
                padded = [-1] * (max_answers - len(start_pos_))
                start_pos_ += padded
                end_pos_ += padded
        return has_answers, start_pos, end_pos
