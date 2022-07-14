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

import datasets

from .wmt_utils import CWMT_SUBSET_NAMES, Wmt, WmtConfig


_PAIRS = [(x, "en") for x in ["de", "ru"]]


class Wmt19(Wmt):
    BUILDER_CONFIGS = [
        WmtConfig(
            description="",
            citation="",
            homepage="",
            license="",
            language_pair=(l1, l2),
            version=datasets.Version("1.0.0"),
        )
        for l1, l2 in _PAIRS
    ]

    @property
    def manual_download_instructions(self):
        if self.config.language_pair[1] in ["cs", "hi", "ru"]:
            return "Please download the data manually as explained. TODO(PVP)"

    @property
    def _subsets(self):
        return {
            datasets.Split.TRAIN: [
                "europarl_v9",
                "europarl_v7_frde",
                "paracrawl_v3",
                "paracrawl_v1_ru",
                "paracrawl_v3_frde",
                "commoncrawl",
                "commoncrawl_frde",
                "newscommentary_v14",
                "newscommentary_v14_frde",
                "czeng_17",
                "yandexcorpus",
                "wikititles_v1",
                "uncorpus_v1",
                "rapid_2016_ltfi",
                "rapid_2019",
            ]
            + CWMT_SUBSET_NAMES,
            datasets.Split.VALIDATION: ["euelections_dev2019", "newsdev2019", "newstest2018"],
        }
