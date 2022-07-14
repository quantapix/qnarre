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
import datasets as ds


_URL = (
    "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/{split}/dataset_{lang}_{split}.json"
)
_LANGS = {
    "de": "German",
    "en": "English",
}

_TRAIN = [_URL.format(split="train", lang=x) for x in _LANGS]
_VALID = [_URL.format(split="dev", lang=x) for x in _LANGS]
_TEST = [_URL.format(split="test", lang=x) for x in _LANGS]


class AmazonReviewsMulti(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name=_LANGS,
            version=ds.Version("1.0.0"),
            languages=_LANGS,
        )
    ] + [
        ds.BuilderConfig(
            name=x,
            version=ds.Version("1.0.0"),
            languages=[x],
        )
        for x in _LANGS
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "review_id": ds.Value("string"),
                    "product_id": ds.Value("string"),
                    "reviewer_id": ds.Value("string"),
                    "stars": ds.Value("int32"),
                    "review_body": ds.Value("string"),
                    "review_title": ds.Value("string"),
                    "language": ds.Value("string"),
                    "product_category": ds.Value("string"),
                }
            ),
        )

    def _split_generators(self, mgr):
        train = mgr.download_and_extract(_TRAIN)
        valid = mgr.download_and_extract(_VALID)
        test = mgr.download_and_extract(_TEST)
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"file_paths": train}),
            ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"file_paths": valid}),
            ds.SplitGenerator(name=ds.Split.TEST, gen_kw={"file_paths": test}),
        ]

    def _generate_examples(self, path):
        i = 0
        for p in path:
            with open(p, "r", encoding="utf-8") as f:
                for x in f:
                    yield i, json.loads(x)
                    i += 1
