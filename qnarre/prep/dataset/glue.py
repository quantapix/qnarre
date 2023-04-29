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

import csv
import os

import numpy as np
import datasets as ds


_MRPC_DEV_IDS = "https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv"
_MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
_MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"

_MNLI_BASE_kw = dict(
    text_features={
        "premise": "sentence1",
        "hypothesis": "sentence2",
    },
    label_classes=["entailment", "neutral", "contradiction"],
    label_column="gold_label",
    data_url="https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    data_dir="MNLI",
)


class Config(ds.BuilderConfig):
    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        label_classes=None,
        process_label=lambda x: x,
        **kw,
    ):
        super(Config, self).__init__(version=ds.Version("1.0.0", ""), **kw)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.process_label = process_label


class Glue(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        Config(
            name="cola",
            text_features={"sentence": "sentence"},
            label_classes=["unacceptable", "acceptable"],
            label_column="is_acceptable",
            data_url="https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
            data_dir="CoLA",
        ),
        Config(
            name="sst2",
            text_features={"sentence": "sentence"},
            label_classes=["negative", "positive"],
            label_column="label",
            data_url="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
            data_dir="SST-2",
        ),
        Config(
            name="mrpc",
            text_features={"sentence1": "", "sentence2": ""},
            label_classes=["not_equivalent", "equivalent"],
            label_column="Quality",
            data_url="",
            data_dir="MRPC",
        ),
        Config(
            name="qqp",
            text_features={
                "question1": "question1",
                "question2": "question2",
            },
            label_classes=["not_duplicate", "duplicate"],
            label_column="is_duplicate",
            data_url="https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
            data_dir="QQP",
        ),
        Config(
            name="stsb",
            text_features={
                "sentence1": "sentence1",
                "sentence2": "sentence2",
            },
            label_column="score",
            data_url="https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
            data_dir="STS-B",
            process_label=np.float32,
        ),
        Config(
            name="mnli",
            **_MNLI_BASE_kw,
        ),
        Config(
            name="mnli_mismatched",
            **_MNLI_BASE_kw,
        ),
        Config(
            name="mnli_matched",
            **_MNLI_BASE_kw,
        ),
        Config(
            name="qnli",
            text_features={
                "question": "question",
                "sentence": "sentence",
            },
            label_classes=["entailment", "not_entailment"],
            label_column="label",
            data_url="https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
            data_dir="QNLI",
        ),
        Config(
            name="rte",
            text_features={
                "sentence1": "sentence1",
                "sentence2": "sentence2",
            },
            label_classes=["entailment", "not_entailment"],
            label_column="label",
            data_url="https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
            data_dir="RTE",
        ),
        Config(
            name="wnli",
            text_features={
                "sentence1": "sentence1",
                "sentence2": "sentence2",
            },
            label_classes=["not_entailment", "entailment"],
            label_column="label",
            data_url="https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
            data_dir="WNLI",
        ),
        Config(
            name="ax",
            text_features={
                "premise": "sentence1",
                "hypothesis": "sentence2",
            },
            label_classes=["entailment", "neutral", "contradiction"],
            label_column="",
            data_url="https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
            data_dir="",
        ),
    ]

    def _info(self):
        fs = {k: ds.Value("string") for k in self.config.text_features.keys()}
        if self.config.label_classes:
            fs["label"] = ds.features.ClassLabel(names=self.config.label_classes)
        else:
            fs["label"] = ds.Value("float32")
        fs["idx"] = ds.Value("int32")
        return ds.DatasetInfo(description="", features=ds.Features(fs), homepage="", citation="")

    def _split_generators(self, mgr):
        if self.config.name == "ax":
            data_file = mgr.download(self.config.data_url)
            return [
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={
                        "data_file": data_file,
                        "split": "test",
                    },
                )
            ]

        if self.config.name == "mrpc":
            data_dir = None
            mrpc_files = mgr.download(
                {
                    "dev_ids": _MRPC_DEV_IDS,
                    "train": _MRPC_TRAIN,
                    "test": _MRPC_TEST,
                }
            )
        else:
            dl_dir = mgr.download_and_extract(self.config.data_url)
            data_dir = os.path.join(dl_dir, self.config.data_dir)
            mrpc_files = None
        train_split = ds.SplitGenerator(
            name=ds.Split.TRAIN,
            gen_kw={
                "data_file": os.path.join(data_dir or "", "train.tsv"),
                "split": "train",
                "mrpc_files": mrpc_files,
            },
        )
        if self.config.name == "mnli":
            return [
                train_split,
                _mnli_split_generator("validation_matched", data_dir, "dev", matched=True),
                _mnli_split_generator("validation_mismatched", data_dir, "dev", matched=False),
                _mnli_split_generator("test_matched", data_dir, "test", matched=True),
                _mnli_split_generator("test_mismatched", data_dir, "test", matched=False),
            ]
        elif self.config.name == "mnli_matched":
            return [
                _mnli_split_generator("validation", data_dir, "dev", matched=True),
                _mnli_split_generator("test", data_dir, "test", matched=True),
            ]
        elif self.config.name == "mnli_mismatched":
            return [
                _mnli_split_generator("validation", data_dir, "dev", matched=False),
                _mnli_split_generator("test", data_dir, "test", matched=False),
            ]
        else:
            return [
                train_split,
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={
                        "data_file": os.path.join(data_dir or "", "dev.tsv"),
                        "split": "dev",
                        "mrpc_files": mrpc_files,
                    },
                ),
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={
                        "data_file": os.path.join(data_dir or "", "test.tsv"),
                        "split": "test",
                        "mrpc_files": mrpc_files,
                    },
                ),
            ]

    def _generate_examples(self, data_file, split, mrpc_files=None):
        if self.config.name == "mrpc":
            examples = self._generate_example_mrpc_files(mrpc_files=mrpc_files, split=split)
            for example in examples:
                yield example["idx"], example
        else:
            process_label = self.config.process_label
            label_classes = self.config.label_classes
            is_cola_non_test = self.config.name == "cola" and split != "test"
            with open(data_file, encoding="utf8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                if is_cola_non_test:
                    reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for n, row in enumerate(reader):
                    if is_cola_non_test:
                        row = {
                            "sentence": row[3],
                            "is_acceptable": row[1],
                        }
                    example = {feat: row[col] for feat, col in self.config.text_features.items()}
                    example["idx"] = n
                    if self.config.label_column in row:
                        label = row[self.config.label_column]
                        if label_classes and label not in label_classes:
                            label = int(label) if label else None
                        example["label"] = process_label(label)
                    else:
                        example["label"] = process_label(-1)
                    for value in example.values():
                        if value is None:
                            break
                    else:
                        yield example["idx"], example

    def _generate_example_mrpc_files(self, mrpc_files, split):
        if split == "test":
            with open(mrpc_files["test"], encoding="utf8") as f:
                f.seek(3)
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for n, row in enumerate(reader):
                    yield {
                        "sentence1": row["#1 String"],
                        "sentence2": row["#2 String"],
                        "label"(row["Quality"]),
                        "idx": n,
                    }
        else:
            with open(mrpc_files["dev_ids"], encoding="utf8") as f:
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                dev_ids = [[row[0], row[1]] for row in reader]
            with open(mrpc_files["train"], encoding="utf8") as f:
                f.seek(3)
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for n, row in enumerate(reader):
                    is_row_in_dev = [row["#1 ID"], row["#2 ID"]] in dev_ids
                    if is_row_in_dev == (split == "dev"):
                        yield {
                            "sentence1": row["#1 String"],
                            "sentence2": row["#2 String"],
                            "label"(row["Quality"]),
                            "idx": n,
                        }


def _mnli_split_generator(name, data_dir, split, matched):
    return ds.SplitGenerator(
        name=name,
        gen_kw={
            "data_file": os.path.join(
                data_dir, "%s_%s.tsv" % (split, "matched" if matched else "mismatched")
            ),
            "split": split,
            "mrpc_files": None,
        },
    )
