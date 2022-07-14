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
import re
import string

from collections import Counter


class Squad(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": {
                        "id": ds.Value("string"),
                        "prediction_text": ds.Value("string"),
                    },
                    "references": {
                        "id": ds.Value("string"),
                        "answers": ds.features.Sequence(
                            {"text": ds.Value("string"), "answer_start": ds.Value("int32")}
                        ),
                    },
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(self, preds, refs):
        ps = {p["id"]: p["prediction_text"] for p in preds}
        x = [{"answers": [{"text": t} for t in r["answers"]["text"]], "id": r["id"]} for r in refs]
        ds = [{"paragraphs": [{"qas": x}]}]
        return _evaluate(ds, ps)


def _evaluate(dset, preds):
    f1 = m = n = 0
    for e in dset:
        for p in e["paragraphs"]:
            for q in p["qas"]:
                n += 1
                i = q["id"]
                if i not in preds:
                    print(f"Missing prediction for {i}")
                    continue
                x = preds[i]
                ts = list(map(lambda t: t["text"], q["answers"]))
                m += _max_over_ys(_match, x, ts)
                f1 += _max_over_ys(_f1, x, ts)
    return {"exact_match": 100.0 * m / n, "f1": 100.0 * f1 / n}


def _max_over_ys(f, x, ts):
    ss = []
    for t in ts:
        ss.append(f(x, t))
    return max(ss)


def _match(x, t):
    return _normalize(x) == _normalize(t)


def _f1(x, t):
    xs = _normalize(x).split()
    ts = _normalize(t).split()
    common = Counter(xs) & Counter(ts)
    s = sum(common.values())
    if s == 0:
        return 0
    precision = 1.0 * s / len(xs)
    recall = 1.0 * s / len(ts)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _normalize(t):
    def no_punc(x):
        exclude = set(string.punctuation)
        return "".join(c for c in x if c not in exclude)

    def no_articles(x):
        return re.sub(r"\b(a|an|the)\b", " ", x)

    def ws_fix(x):
        return " ".join(x.split())

    return ws_fix(no_articles(no_punc(t.lower())))
