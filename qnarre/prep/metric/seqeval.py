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

import importlib
import datasets as ds

from seqeval.metrics import accuracy_score, classification_report


class Seqeval(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            homepage="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Sequence(ds.Value("string", id="label"), id="sequence"),
                    "references": ds.Sequence(ds.Value("string", id="label"), id="sequence"),
                }
            ),
        )

    def _compute(
        self,
        preds,
        refs,
        suffix=False,
        scheme=None,
        mode=None,
        sample_weight=None,
        zero_division="warn",
    ):
        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError()
        report = classification_report(
            y_true=refs,
            y_pred=preds,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")
        y = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in report.items()
        }
        y["overall_precision"] = overall_score["precision"]
        y["overall_recall"] = overall_score["recall"]
        y["overall_f1"] = overall_score["f1-score"]
        y["overall_accuracy"] = accuracy_score(y_true=refs, y_pred=preds)
        return y
