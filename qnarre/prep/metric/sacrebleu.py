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

import sacrebleu as scb
import datasets as ds


class Sacrebleu(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            homepage="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Value("string", id="sequence"),
                    "references": ds.Sequence(ds.Value("string", id="sequence"), id="references"),
                }
            ),
        )

    def _compute(
        self,
        preds,
        refs,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=None,
        use_effective_order=False,
    ):
        references_per_prediction = len(refs[0])
        if any(len(refs) != references_per_prediction for refs in refs):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")
        transformed_references = [
            [refs[i] for refs in refs] for i in range(references_per_prediction)
        ]
        y = scb.corpus_bleu(
            preds,
            transformed_references,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            force=force,
            lowercase=lowercase,
            use_effective_order=use_effective_order,
            **(dict(tokenize=tokenize) if tokenize else {}),
        )
        return {
            "score": y.score,
            "counts": y.counts,
            "totals": y.totals,
            "precisions": y.precisions,
            "bp": y.bp,
            "sys_len": y.sys_len,
            "ref_len": y.ref_len,
        }
