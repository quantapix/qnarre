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

from datasets.tasks import QuestionAnsweringExtractive

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": _URL + "train-v2.0.json",
    "valid": _URL + "dev-v2.0.json",
}


class Squad2(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name="squad2", version=ds.Version("2.0.0"))]

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
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, mgr):
        fs = mgr.download_and_extract(_URLS)
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"filepath": fs["train"]}),
            ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"filepath": fs["valid"]}),
        ]

    def _generate_examples(self, path):
        with open(path, encoding="utf-8") as f:
            for e in json.load(f)["data"]:
                t = e.get("title", "")
                for p in e["paragraphs"]:
                    c = p["context"]
                    for q in p["qas"]:
                        i = q["id"]
                        ss = [a["answer_start"] for a in q["answers"]]
                        xs = [a["text"] for a in q["answers"]]
                        yield i, {
                            "title": t,
                            "context": c,
                            "question": q["question"],
                            "id": i,
                            "answers": {"answer_start": ss, "text": xs},
                        }
