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

import bert_score
import datasets as ds
import functools

from contextlib import contextmanager
from packaging import version


@contextmanager
def filter_logging_context():
    def filter_log(record):
        return False if "This IS expected if you are initializing" in record.msg else True

    logger = ds.utils.logging.get_logger("transformers.modeling_utils")
    logger.addFilter(filter_log)
    try:
        yield
    finally:
        logger.removeFilter(filter_log)


class BERTScore(ds.Metric):
    def _info(self):
        return ds.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=ds.Features(
                {
                    "predictions": ds.Value("string", id="sequence"),
                    "references": ds.Sequence(ds.Value("string", id="sequence"), id="references"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(
        self,
        predictions,
        references,
        lang=None,
        model_type=None,
        n_lays=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
    ):
        get_hash = bert_score.utils.get_hash
        scorer = bert_score.BERTScorer
        if version.parse(bert_score.__version__) >= version.parse("0.3.10"):
            get_hash = functools.partial(get_hash, use_fast_tokenizer=use_fast_tokenizer)
            scorer = functools.partial(scorer, use_fast_tokenizer=use_fast_tokenizer)
        elif use_fast_tokenizer:
            raise ImportWarning(
                "To use a fast tokenizer, the module `bert-score>=0.3.10` is required, and the current version of `bert-score` doesn't match this condition.\n"
                'You can install it with `pip install "bert-score>=0.3.10"`.'
            )
        if model_type is None:
            assert lang is not None, "either lang or model_type should be specified"
            model_type = bert_score.utils.lang2model[lang.lower()]
        if n_lays is None:
            n_lays = bert_score.utils.model2layers[model_type]
        hashcode = get_hash(
            model=model_type,
            n_lays=n_lays,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            use_custom_baseline=baseline_path is not None,
        )
        with filter_logging_context():
            if not hasattr(self, "cached_bertscorer") or self.cached_bertscorer.hash != hashcode:
                self.cached_bertscorer = scorer(
                    model_type=model_type,
                    n_lays=n_lays,
                    batch_size=batch_size,
                    nthreads=nthreads,
                    all_layers=all_layers,
                    idf=idf,
                    device=device,
                    lang=lang,
                    rescale_with_baseline=rescale_with_baseline,
                    baseline_path=baseline_path,
                )

        (P, R, F) = self.cached_bertscorer.score(
            cands=predictions,
            refs=references,
            verbose=verbose,
            batch_size=batch_size,
        )
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F.tolist(),
            "hashcode": hashcode,
        }

    def add_batch(self, preds=None, refs=None, **kw):
        if refs is not None:
            refs = [[r] if isinstance(r, str) else r for r in refs]
        super().add_batch(preds, refs, **kw)

    def add(self, pred=None, ref=None, **kw):
        if isinstance(ref, str):
            ref = [ref]
        super().add(pred, ref, **kw)
