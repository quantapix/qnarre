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

_URL = "https://github.com/ZihanWangKi/CrossWeigh/raw/master/data/"
_URLS = {
    "train": f"{_URL}conllpp_train.txt",
    "valid": f"{_URL}conllpp_dev.txt",
    "test": f"{_URL}conllpp_test.txt",
}


class Conllpp(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name="conllpp", version=ds.Version("1.0.0"))]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "id": ds.Value("string"),
                    "tokens": ds.Sequence(ds.Value("string")),
                    "pos_tags": ds.Sequence(
                        ds.features.ClassLabel(
                            names=[
                                '"',
                                "''",
                                "#",
                                "$",
                                "(",
                                ")",
                                ",",
                                ".",
                                ":",
                                "``",
                                "CC",
                                "CD",
                                "DT",
                                "EX",
                                "FW",
                                "IN",
                                "JJ",
                                "JJR",
                                "JJS",
                                "LS",
                                "MD",
                                "NN",
                                "NNP",
                                "NNPS",
                                "NNS",
                                "NN|SYM",
                                "PDT",
                                "POS",
                                "PRP",
                                "PRP$",
                                "RB",
                                "RBR",
                                "RBS",
                                "RP",
                                "SYM",
                                "TO",
                                "UH",
                                "VB",
                                "VBD",
                                "VBG",
                                "VBN",
                                "VBP",
                                "VBZ",
                                "WDT",
                                "WP",
                                "WP$",
                                "WRB",
                            ]
                        )
                    ),
                    "chunk_tags": ds.Sequence(
                        ds.features.ClassLabel(
                            names=[
                                "O",
                                "B-ADJP",
                                "I-ADJP",
                                "B-ADVP",
                                "I-ADVP",
                                "B-CONJP",
                                "I-CONJP",
                                "B-INTJ",
                                "I-INTJ",
                                "B-LST",
                                "I-LST",
                                "B-NP",
                                "I-NP",
                                "B-PP",
                                "I-PP",
                                "B-PRT",
                                "I-PRT",
                                "B-SBAR",
                                "I-SBAR",
                                "B-UCP",
                                "I-UCP",
                                "B-VP",
                                "I-VP",
                            ]
                        )
                    ),
                    "ner_tags": ds.Sequence(
                        ds.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                                "B-MISC",
                                "I-MISC",
                            ]
                        )
                    ),
                }
            ),
        )

    def _split_generators(self, mgr):
        fs = mgr.download_and_extract(_URLS)
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"filepath": fs["train"]}),
            ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kw={"filepath": fs["dev"]}),
            ds.SplitGenerator(name=ds.Split.TEST, gen_kw={"filepath": fs["test"]}),
        ]

    def _generate_examples(self, path):
        with open(path, encoding="utf-8") as f:
            i = 0
            ts = []
            pos = []
            chunks = []
            ners = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if ts:
                        yield i, {
                            "id"(i),
                            "tokens": ts,
                            "pos_tags": pos,
                            "chunk_tags": chunks,
                            "ner_tags": ners,
                        }
                        i += 1
                        ts = []
                        pos = []
                        chunks = []
                        ners = []
                else:
                    splits = line.split(" ")
                    ts.append(splits[0])
                    pos.append(splits[1])
                    chunks.append(splits[2])
                    ners.append(splits[3].rstrip())
            if ts:
                yield i, {
                    "id"(i),
                    "tokens": ts,
                    "pos_tags": pos,
                    "chunk_tags": chunks,
                    "ner_tags": ners,
                }
