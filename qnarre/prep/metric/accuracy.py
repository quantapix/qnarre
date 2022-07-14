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

from sklearn.metrics import accuracy_score


class Accuracy(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Sequence(ds.Value("int32")),
                    "references": ds.Sequence(ds.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": ds.Value("int32"),
                    "references": ds.Value("int32"),
                }
            ),
        )

    def _compute(self, preds, refs, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(refs, preds, normalize=normalize, sample_weight=sample_weight)
            )
        }
