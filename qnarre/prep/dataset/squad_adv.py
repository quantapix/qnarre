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

_URL = "https://worksheets.codalab.org/rest/bundles/"
_URLS = {
    "AddSent": _URL + "0xb765680b60c64d088f5daccac08b3905/contents/blob/",
    "AddOneSent": _URL + "0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/",
}


class SquadAdversarial(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(name="AddSent", version=ds.Version("1.1.0")),
        ds.BuilderConfig(name="AddOneSent", version=ds.Version("1.1.0")),
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "id": ds.Value("string"),
                    "title": ds.Value("string"),
                    "context": ds.Value("string"),
                    "question": ds.Value("string"),
                    "answers": ds.features.Sequence(
                        {"text": ds.Value("string"), "answer_start": ds.Value("int32")}
                    ),
                }
            ),
        )

    def _split_generators(self, mgr):
        fs = mgr.download_and_extract(_URLS)
        n = self.config.name
        return [ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"filepath": fs[n]})]

    def _generate_examples(self, path):
        with open(path, encoding="utf-8") as f:
            for e in json.load(f)["data"]:
                t = e.get("title", "").strip()
                for p in e["paragraphs"]:
                    c = p["context"].strip()
                    for q in p["qas"]:
                        i = q["id"]
                        ss = [a["answer_start"] for a in q["answers"]]
                        xs = [a["text"].strip() for a in q["answers"]]
                        yield i, {
                            "title": t,
                            "context": c,
                            "question": q["question"].strip(),
                            "id": i,
                            "answers": {"answer_start": ss, "text": xs},
                        }
