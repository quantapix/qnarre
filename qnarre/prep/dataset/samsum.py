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

import json
import py7zr

import datasets as ds

_URLS = "https://arxiv.org/src/1911.12237v2/anc/corpus.7z"


class Samsum(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name="samsum", version=ds.Version("1.1.0"))]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "id": ds.Value("string"),
                    "dialogue": ds.Value("string"),
                    "summary": ds.Value("string"),
                }
            ),
        )

    def _split_generators(self, mgr):
        path = mgr.download_and_extract(_URLS)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={"filepath": (path, "train.json"), "split": "train"},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={"filepath": (path, "test.json"), "split": "test"},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={"filepath": (path, "val.json"), "split": "val"},
            ),
        ]

    def _generate_examples(self, filepath, _):
        path, fname = filepath
        with py7zr.SevenZipFile(path, "r") as z:
            for name, bio in z.readall().items():
                if name == fname:
                    data = json.load(bio)
        for e in data:
            yield e["id"], e
