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

import datasets as ds
import numpy as np
import pickle

from datasets.tasks import ImageClassification


_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Cifar10(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name="plain_text", version=ds.Version("1.0.0"))]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "img": ds.Image(),
                    "label": ds.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("img", "label"),
            task_templates=ImageClassification(image_column="img", label_column="label"),
        )

    def _split_generators(self, mgr):
        fs = mgr.download(_URL)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={"files": mgr.iter_archive(fs), "split": "train"},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={"files": mgr.iter_archive(fs), "split": "test"},
            ),
        ]

    def _generate_examples(self, fs, split):
        if split == "train":
            batches = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        if split == "test":
            batches = ["test_batch"]
        batches = [f"cifar-10-batches-py/{p}" for p in batches]
        for p, f in fs:
            if p in batches:
                d = pickle.load(f, encoding="bytes")
                xs = d[b"data"]
                ls = d[b"labels"]
                for i, _ in enumerate(xs):
                    x = np.transpose(np.reshape(xs[i], (3, 32, 32)), (1, 2, 0))
                    yield f"{p}_{i}", {"img": x, "label": ls[i]}
