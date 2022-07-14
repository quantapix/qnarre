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

from rouge_score import rouge_scorer, scoring


class Rouge(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Value("string", id="sequence"),
                    "references": ds.Value("string", id="sequence"),
                }
            ),
        )

    def _compute(self, preds, refs, rouge_types=None, use_agregator=True, use_stemmer=False):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_agregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []
        for r, p in zip(refs, preds):
            score = scorer.score(r, p)
            if use_agregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)
        if use_agregator:
            y = aggregator.aggregate()
        else:
            y = {}
            for k in scores[0]:
                y[k] = list(score[k] for score in scores)
        return y
