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


_ID = "id"
_LINK = "link"
_TITLE = "title"
_ARTICLE = "article"
_HIGHLIGHTS = "highlights"

_TRAIN = "https://drive.google.com/u/0/uc?id=1-CaP3xHgZxOGjQ3pXC5tr9YnIajmel-t&export=download"
_TEST = "https://drive.google.com/u/0/uc?id=1-9G4yYP6YO8oMA-o4cTe9NJpEyr7x5jg&export=download"
_VALID = "https://drive.google.com/u/0/uc?id=1-2g2gkDeNaN-vth-8Mgit_ovmSkVh91u&export=download"


class WikiSummary(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.1.0")

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {k: ds.Value("string") for k in [_ID, _LINK, _TITLE, _ARTICLE, _HIGHLIGHTS]}
            ),
        )

    def _split_generators(self, mgr):
        train = mgr.download_and_extract(_TRAIN)
        test = mgr.download_and_extract(_TEST)
        valid = mgr.download_and_extract(_VALID)
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"filepath": train}),
            ds.SplitGenerator(name=ds.Split.TEST, gen_kw={"filepath": test}),
            ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"filepath": valid}),
        ]

    def _generate_examples(self, path):
        with open(path, encoding="utf8") as f:
            r = csv.reader(
                f,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            for i, row in enumerate(r):
                if len(row) == 5:
                    yield i, {
                        _ID: row[0],
                        _LINK: row[1],
                        _TITLE: row[2],
                        _ARTICLE: row[3],
                        _HIGHLIGHTS: row[4],
                    }
