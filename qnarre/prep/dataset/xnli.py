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

from os.path import join


_URL = "https://dl.fbaipublicfiles.com/XNLI/"
_URLS = {
    "train": _URL + "XNLI-MT-1.0.zip",
    "valid": _URL + "XNLI-1.0.zip",
}

_LANGS = ("de", "en")


class Xnli(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name=x, version=ds.Version("1.1.0")) for x in _LANGS]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "premise": ds.Value("string"),
                    "hypothesis": ds.Value("string"),
                    "label": ds.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                }
            ),
        )

    def _split_generators(self, mgr):
        fs = mgr.download_and_extract(_URLS)
        t = join(fs["train"], "XNLI-MT-1.0", "multinli")
        v = join(fs["valid"], "XNLI-1.0")
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={
                    "filepaths": join(t, f"multinli.train.{self.config.name}.tsv"),
                    "data_format": "XNLI-MT",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={"filepaths": [join(v, "xnli.test.tsv")], "data_format": "XNLI"},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={"filepaths": [join(v, "xnli.dev.tsv")], "data_format": "XNLI"},
            ),
        ]

    def _generate_examples(self, fmt, fs):
        if fmt == "XNLI-MT":
            for i, path in enumerate(fs):
                f = open(path, encoding="utf-8")
                r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for j, x in enumerate(r):
                    k = str(i) + "_" + str(j)
                    yield k, {
                        "premise": x["premise"],
                        "hypothesis": x["hypo"],
                        "label": x["label"].replace("contradictory", "contradiction"),
                    }
        else:
            for path in fs:
                with open(path, encoding="utf-8") as f:
                    r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for x in r:
                        if x["language"] == self.config.name:
                            yield x["pairID"], {
                                "premise": x["sentence1"],
                                "hypothesis": x["sentence2"],
                                "label": x["gold_label"],
                            }
