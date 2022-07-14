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

import collections
import datasets as ds
import re
import string

from collections import Counter


class Squad2(ds.Metric):
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
                        "no_answer_probability": ds.Value("float32"),
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

    def _compute(self, preds, refs, no_answer_threshold=1.0):
        probs = {p["id"]: p["no_answer_probability"] for p in preds}
        preds = {p["id"]: p["prediction_text"] for p in preds}
        ds = [{"paragraphs": [{"qas": refs}]}]
        has_ans = _make_map(ds)
        ans_ids = [k for k, v in has_ans.items() if v]
        no_ids = [k for k, v in has_ans.items() if not v]
        ms_raw, f1_raw = _raw_scores(ds, preds)
        ms = _apply(ms_raw, probs, has_ans, no_answer_threshold)
        f1 = _apply(f1_raw, probs, has_ans, no_answer_threshold)
        ys = _eval(ms, f1)
        if ans_ids:
            _merge(ys, _eval(ms, f1, ans_ids), "HasAns")
        if no_ids:
            _merge(ys, _eval(ms, f1, no_ids), "NoAns")
        _best_thresh(ys, preds, ms_raw, f1_raw, probs, has_ans)
        return dict(ys)


OPTS = None


def _make_map(ds):
    ys = {}
    for e in ds:
        for p in e["paragraphs"]:
            for x in p["qas"]:
                ys[x["id"]] = bool(x["answers"]["text"])
    return ys


def _raw_scores(ds, preds):
    ms = {}
    f1 = {}
    for e in ds:
        for p in e["paragraphs"]:
            for q in p["qas"]:
                i = q["id"]
                if i not in preds:
                    print(f"Missing prediction for {i}")
                    continue
                x = preds[i]
                ts = [t for t in q["answers"]["text"] if _normalize(t)]
                ts = ts if ts else [""]
                ms[i] = max(_match(x, t) for t in ts)
                f1[i] = max(_f1(x, t) for t in ts)
    return ms, f1


def _match(x, t):
    return int(_normalize(x) == _normalize(t))


def _f1(x, t):
    xs = _normalize(x).split() if x else []
    ts = _normalize(t).split() if t else []
    if len(xs) == 0 or len(ts) == 0:
        return int(xs == ts)
    common = Counter(ts) & Counter(xs)
    s = sum(common.values())
    if s == 0:
        return 0
    precision = 1.0 * s / len(xs)
    recall = 1.0 * s / len(ts)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _apply(scores, probs, has_ans, thresh):
    ys = {}
    for i, s in scores.items():
        if probs[i] > thresh:
            ys[i] = float(not has_ans[i])
        else:
            ys[i] = s
    return ys


def _eval(ms, f1, ids):
    if not ids:
        n = len(ms)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(ms.values()) / n),
                ("f1", 100.0 * sum(f1.values()) / n),
                ("total", n),
            ]
        )
    else:
        n = len(ids)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(ms[i] for i in ids) / n),
                ("f1", 100.0 * sum(f1[i] for i in ids) / n),
                ("total", n),
            ]
        )


def _merge(ys, xs, pre):
    for x in xs:
        ys[f"{pre}_{x}"] = xs[x]


def _best_thresh(ys, preds, ms_raw, f1_raw, probs, has_ans):
    ms, m_thresh = _find_best(preds, ms_raw, probs, has_ans)
    f1, f1_thresh = _find_best(preds, f1_raw, probs, has_ans)
    ys["best_exact"] = ms
    ys["best_exact_thresh"] = m_thresh
    ys["best_f1"] = f1
    ys["best_f1_thresh"] = f1_thresh


def _find_best(preds, scores, probs, has_ans):
    y = x = sum(1 for k in has_ans if not has_ans[k])
    t = 0.0
    ids = sorted(probs, key=lambda k: probs[k])
    for i in ids:
        if i not in scores:
            continue
        if has_ans[i]:
            d = scores[i]
        else:
            d = -1 if preds[i] else 0
        x += d
        if x > y:
            y = x
            t = probs[i]
    return 100.0 * y / len(scores), t


def _normalize(t):
    def no_punc(x):
        exclude = set(string.punctuation)
        return "".join(c for c in x if c not in exclude)

    def no_articles(x):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", x)

    def ws_fix(x):
        return " ".join(x.split())

    return ws_fix(no_articles(no_punc(t.lower())))
