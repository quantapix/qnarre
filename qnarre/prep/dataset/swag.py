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

import datasets as ds

_URLs = {
    "full": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train_full.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val_full.csv",
    },
    "regular": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val.csv",
        "test": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/test.csv",
    },
}


class Swag(ds.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ds.BuilderConfig(name="regular", version=ds.Version("1.1.0")),
        ds.BuilderConfig(name="full", version=ds.Version("1.1.0")),
    ]

    DEFAULT_CONFIG_NAME = "regular"

    def _info(self):
        if self.config.name == "regular":
            features = ds.Features(
                {
                    "video-id": ds.Value("string"),
                    "fold-ind": ds.Value("string"),
                    "startphrase": ds.Value("string"),
                    "sent1": ds.Value("string"),
                    "sent2": ds.Value("string"),
                    "gold-source": ds.Value("string"),
                    "ending0": ds.Value("string"),
                    "ending1": ds.Value("string"),
                    "ending2": ds.Value("string"),
                    "ending3": ds.Value("string"),
                    "label": ds.ClassLabel(names=["0", "1", "2", "3"]),
                }
            )
        else:
            features = ds.Features(
                {
                    "video-id": ds.Value("string"),
                    "fold-ind": ds.Value("string"),
                    "startphrase": ds.Value("string"),
                    "gold-ending": ds.Value("string"),
                    "distractor-0": ds.Value("string"),
                    "distractor-1": ds.Value("string"),
                    "distractor-2": ds.Value("string"),
                    "distractor-3": ds.Value("string"),
                    "gold-source": ds.Value("string"),
                    "gold-type": ds.Value("string"),
                    "distractor-0-type": ds.Value("string"),
                    "distractor-1-type": ds.Value("string"),
                    "distractor-2-type": ds.Value("string"),
                    "distractor-3-type": ds.Value("string"),
                    "sent1": ds.Value("string"),
                    "sent2": ds.Value("string"),
                }
            )
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=features,
        )

    def _split_generators(self, mgr):
        fs = mgr.download_and_extract(_URLs[self.config.name])
        splits = [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={"filepath": fs["train"], "split": "train"},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={"filepath": fs["val"], "split": "val"},
            ),
        ]
        if self.config.name == "regular":
            splits.append(
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"filepath": fs["test"], "split": "test"},
                )
            )
        return splits

    def _generate_examples(self, filepath, split):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = list(csv.reader(f, delimiter=","))
            for i, row in enumerate(lines[1:]):
                if self.config.name == "regular":
                    yield i, {
                        "video-id": row[1],
                        "fold-ind": row[2],
                        "startphrase": row[3],
                        "sent1": row[4],
                        "sent2": row[5],
                        "gold-source": row[6],
                        "ending0": row[7],
                        "ending1": row[8],
                        "ending2": row[9],
                        "ending3": row[10],
                        "label": -1 if split == "test" else row[11],
                    }
                else:
                    yield i, {
                        "video-id": row[0],
                        "fold-ind": row[1],
                        "startphrase": row[2],
                        "gold-ending": row[3],
                        "distractor-0": row[4],
                        "distractor-1": row[5],
                        "distractor-2": row[6],
                        "distractor-3": row[7],
                        "gold-source": row[8],
                        "gold-type": row[9],
                        "distractor-0-type": row[10],
                        "distractor-1-type": row[11],
                        "distractor-2-type": row[12],
                        "distractor-3-type": row[13],
                        "sent1": row[14],
                        "sent2": row[15],
                    }
