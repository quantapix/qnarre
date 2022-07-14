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

from os.path import join

_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
_URLS = {
    "103-v1": _URL + "wikitext-103-v1.zip",
    "2-v1": _URL + "wikitext-2-v1.zip",
    "103-raw-v1": _URL + "wikitext-103-raw-v1.zip",
    "2-raw-v1": _URL + "wikitext-2-raw-v1.zip",
}


class Wikitext(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(name="103-v1", version=ds.Version("0.1.0")),
        ds.BuilderConfig(name="2-v1", version=ds.Version("0.1.0")),
        ds.BuilderConfig(name="103-raw-v1", version=ds.Version("0.1.0")),
        ds.BuilderConfig(name="2-raw-v1", version=ds.Version("0.1.0")),
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features({"text": ds.Value("string")}),
        )

    def _split_generators(self, mgr):
        n = self.config.name
        f = mgr.download_and_extract(_URLS[n])
        if n == "103-v1":
            d = join(f, "wikitext-103")
            return [
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"data_file": join(d, "wiki.test.tokens"), "split": "test"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"data_file": join(d, "wiki.train.tokens"), "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"data_file": join(d, "wiki.valid.tokens"), "split": "valid"},
                ),
            ]
        if n == "wikitext-103-raw-v1":
            d = join(f, "wikitext-103-raw")
            return [
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"data_file": join(d, "wiki.test.raw"), "split": "test"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"data_file": join(d, "wiki.train.raw"), "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"data_file": join(d, "wiki.valid.raw"), "split": "valid"},
                ),
            ]
        if n == "wikitext-2-raw-v1":
            d = join(f, "wikitext-2-raw")
            return [
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"data_file": join(d, "wiki.test.raw"), "split": "test"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"data_file": join(d, "wiki.train.raw"), "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"data_file": join(d, "wiki.valid.raw"), "split": "valid"},
                ),
            ]
        if n == "wikitext-2-v1":
            d = join(f, "wikitext-2")
            return [
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"data_file": join(d, "wiki.test.tokens"), "split": "test"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"data_file": join(d, "wiki.train.tokens"), "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"data_file": join(d, "wiki.valid.tokens"), "split": "valid"},
                ),
            ]

    def _generate_examples(self, x, _):
        with open(x, encoding="utf-8") as f:
            for i, t in enumerate(f):
                if t.strip():
                    yield i, {"text": t}
                else:
                    yield i, {"text": ""}
