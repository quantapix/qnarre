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
import os

import datasets as ds

_URL_DATA = "http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz"
_URL_SPLITS = "https://raw.githubusercontent.com/EdinburghNLP/XSum/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"

_DOC = "document"
_SUM = "summary"
_ID = "id"

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)


class Xsum(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.2.0")

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    _DOC: ds.Value("string"),
                    _SUM: ds.Value("string"),
                    _ID: ds.Value("string"),
                }
            ),
            supervised_keys=(_DOC, _SUM),
        )

    def _split_generators(self, mgr):
        fs = mgr.download({"data": _URL_DATA, "splits": _URL_SPLITS})
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={
                    "split_path": fs["splits"],
                    "split_name": "train",
                    "data_dir": "bbc-summary-data",
                    "files": mgr.iter_archive(fs["data"]),
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={
                    "split_path": fs["splits"],
                    "split_name": "validation",
                    "data_dir": "bbc-summary-data",
                    "files": mgr.iter_archive(fs["data"]),
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={
                    "split_path": fs["splits"],
                    "split_name": "test",
                    "data_dir": "bbc-summary-data",
                    "files": mgr.iter_archive(fs["data"]),
                },
            ),
        ]

    def _generate_examples(self, split_path, split_name, data_dir, files):
        with open(split_path, "r", encoding="utf-8") as f:
            split_ids = json.load(f)
        split_ids = {k: set(v) for k, v in split_ids.items()}
        for path, f in files:
            if not split_ids[split_name]:
                break
            elif path.startswith(data_dir) and path.endswith(".summary"):
                i = os.path.basename(path).split(".")[0]
                if i in split_ids[split_name]:
                    split_ids[split_name].remove(i)
                    text = "".join(
                        [
                            line.decode("utf-8")
                            for line in f.readlines()
                            if line.decode("utf-8") not in _REMOVE_LINES and line.strip()
                        ]
                    )
                    segs = text.split("[SN]")
                    yield i, {_DOC: segs[8].strip(), _SUM: segs[6].strip(), _ID: i}
