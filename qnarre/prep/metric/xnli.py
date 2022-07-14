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


class Xnli(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Value("int64" if self.config_name != "sts-b" else "float32"),
                    "references": ds.Value("int64" if self.config_name != "sts-b" else "float32"),
                }
            ),
            format="numpy",
        )

    def _compute(self, preds, refs):
        return {"accuracy": _accuracy(preds, refs)}


def _accuracy(preds, labels):
    return (preds == labels).mean()
