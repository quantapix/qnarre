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

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef


class Glue(ds.Metric):
    def _info(self):
        assert self.config_name in [
            "sst2",
            "mnli",
            "mnli_mismatched",
            "mnli_matched",
            "cola",
            "stsb",
            "mrpc",
            "qqp",
            "qnli",
            "rte",
            "wnli",
            "hans",
        ]
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Value("int64" if self.config_name != "stsb" else "float32"),
                    "references": ds.Value("int64" if self.config_name != "stsb" else "float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, preds, refs):
        if self.config_name == "cola":
            return {"matthews_correlation": matthews_corrcoef(refs, preds)}
        elif self.config_name == "stsb":
            return _pearson_and_spearman(preds, refs)
        elif self.config_name in ["mrpc", "qqp"]:
            return _acc_and_f1(preds, refs)
        else:
            assert self.config_name in [
                "sst2",
                "mnli",
                "mnli_mismatched",
                "mnli_matched",
                "qnli",
                "rte",
                "wnli",
                "hans",
            ]
            return {"accuracy": _accuracy(preds, refs)}


def _accuracy(preds, refs):
    return float((preds == refs).mean())


def _acc_and_f1(preds, refs):
    acc = _accuracy(preds, refs)
    f1 = float(f1_score(y_true=refs, y_pred=preds))
    return {"accuracy": acc, "f1": f1}


def _pearson_and_spearman(preds, refs):
    p = float(pearsonr(preds, refs)[0])
    s = float(spearmanr(preds, refs)[0])
    return {"pearson": p, "spearmanr": s}
