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

import glob
import gzip
import json
import os

import datasets as ds


_URL = "https://drive.google.com/uc?export=download&id=1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"

_DOC = "description"
_SUM = "abstract"

_CPC = {
    "a": "Human Necessities",
    "b": "Performing Operations; Transporting",
    "c": "Chemistry; Metallurgy",
    "d": "Textiles; Paper",
    "e": "Fixed Constructions",
    "f": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    "g": "Physics",
    "h": "Electricity",
    "y": "General tagging of new or cross-sectional technology",
}


class Config(ds.BuilderConfig):
    def __init__(self, *xs, cpc=None, **kw):
        super().__init__(*xs, version=ds.Version("1.0.0"), **kw)
        self.cpc = cpc


class BigPatent(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [Config(cpc=list(_CPC), name="all")] + [
        Config(cpc=[k], name=k) for k, v in sorted(_CPC.items())
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features({_DOC: ds.Value("string"), _SUM: ds.Value("string")}),
            supervised_keys=(_DOC, _SUM),
        )

    def _split_generators(self, mgr):
        p = mgr.download_and_extract(_URL)
        ks = ["train", "valid", "test"]
        fs = mgr.extract({k: os.path.join(p, "bigPatentData", k + ".tar.gz") for k in ks})
        fs = {k: os.path.join(fs[k], k) for k in ks}
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"path": fs["train"]}),
            ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"path": fs["val"]}),
            ds.SplitGenerator(name=ds.Split.TEST, gen_kw={"path": fs["test"]}),
        ]

    def _generate_examples(self, path=None):
        for c in self.config.cpc:
            ns = glob.glob(os.path.join(path, c, "*"))
            for n in sorted(ns):
                with open(n, "rb") as f:
                    f = gzip.GzipFile(fileobj=f)
                    for r in f:
                        x = json.loads(r)
                        yield x["publication_number"], {_DOC: x[_DOC], _SUM: x[_SUM]}
