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

import json
import os

import datasets as ds

_XGLUE_ALL_DATA = "https://xglue.blob.core.windows.net/xglue/xglue_full_dataset.tar.gz"

_LANGS = {
    "ner": ["en", "de", "es", "nl"],
    "pos": ["en", "de"],
    "mlqa": ["en", "de"],
    "nc": ["en", "de"],
    "xnli": ["en", "de"],
    "paws-x": ["en", "de"],
    "qadsm": ["en", "de"],
    "wpr": ["en", "de"],
    "qam": ["en", "de"],
    "qg": ["en", "de"],
    "ntg": ["en", "de"],
}

_PATHS = {
    "mlqa": {
        "train": os.path.join("squad1.1", "train-v1.1.json"),
        "dev": os.path.join("MLQA_V1", "dev", "dev-context-{0}-question-{0}.json"),
        "test": os.path.join("MLQA_V1", "test", "test-context-{0}-question-{0}.json"),
    },
    "xnli": {"train": "multinli.train.en.tsv", "dev": "{}.dev", "test": "{}.test"},
    "paws-x": {
        "train": os.path.join("en", "train.tsv"),
        "dev": os.path.join("{}", "dev_2k.tsv"),
        "test": os.path.join("{}", "test_2k.tsv"),
    },
}
for x in ["ner", "pos"]:
    _PATHS[x] = {"train": "en.train", "dev": "{}.dev", "test": "{}.test"}
for x in ["nc", "qadsm", "wpr", "qam"]:
    _PATHS[x] = {
        "train": "xglue." + x + ".en.train",
        "dev": "xglue." + x + ".{}.dev",
        "test": "xglue." + x + ".{}.test",
    }
for x in ["qg", "ntg"]:
    _PATHS[x] = {
        "train": "xglue." + x + ".en",
        "dev": "xglue." + x + ".{}",
        "test": "xglue." + x + ".{}",
    }


class Config(ds.BuilderConfig):
    def __init__(
        self,
        data_dir,
        citation,
        url,
        **kw,
    ):
        super(Config, self).__init__(version=ds.Version("1.0.0", ""), **kw)
        self.data_dir = data_dir
        self.citation = citation
        self.url = url


class XGlue(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        Config(name="ner", data_dir="NER"),
        Config(name="pos", data_dir="POS"),
        Config(name="mlqa", data_dir="MLQA"),
        Config(name="nc", data_dir="NC"),
        Config(name="xnli", data_dir="XNLI"),
        Config(name="paws-x", data_dir="PAWSX"),
        Config(name="qadsm", data_dir="QADSM"),
        Config(name="wpr", data_dir="WPR"),
        Config(name="qam", data_dir="QAM"),
        Config(name="qg", data_dir="QG"),
        Config(name="ntg", data_dir="NTG"),
    ]

    def _info(self):
        if self.config.name == "ner":
            features = {
                "words": ds.Sequence(ds.Value("string")),
                "ner": ds.Sequence(
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
        elif self.config.name == "pos":
            features = {
                "words": ds.Sequence(ds.Value("string")),
                "pos": ds.Sequence(
                    ds.features.ClassLabel(
                        names=[
                            "ADJ",
                            "ADP",
                            "ADV",
                            "AUX",
                            "CCONJ",
                            "DET",
                            "INTJ",
                            "NOUN",
                            "NUM",
                            "PART",
                            "PRON",
                            "PROPN",
                            "PUNCT",
                            "SCONJ",
                            "SYM",
                            "VERB",
                            "X",
                        ]
                    )
                ),
            }
        elif self.config.name == "mlqa":
            features = {
                "context": ds.Value("string"),
                "question": ds.Value("string"),
                "answers": ds.features.Sequence(
                    {"answer_start": ds.Value("int32"), "text": ds.Value("string")}
                ),
            }
        elif self.config.name == "nc":
            features = {
                "news_title": ds.Value("string"),
                "news_body": ds.Value("string"),
                "news_category": ds.ClassLabel(
                    names=[
                        "foodanddrink",
                        "sports",
                        "travel",
                        "finance",
                        "lifestyle",
                        "news",
                        "entertainment",
                        "health",
                        "video",
                        "autos",
                    ]
                ),
            }
        elif self.config.name == "xnli":
            features = {
                "premise": ds.Value("string"),
                "hypothesis": ds.Value("string"),
                "label": ds.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
            }
        elif self.config.name == "paws-x":
            features = {
                "sentence1": ds.Value("string"),
                "sentence2": ds.Value("string"),
                "label": ds.features.ClassLabel(names=["different", "same"]),
            }
        elif self.config.name == "qadsm":
            features = {
                "query": ds.Value("string"),
                "ad_title": ds.Value("string"),
                "ad_description": ds.Value("string"),
                "relevance_label": ds.features.ClassLabel(names=["Bad", "Good"]),
            }
        elif self.config.name == "wpr":
            features = {
                "query": ds.Value("string"),
                "web_page_title": ds.Value("string"),
                "web_page_snippet": ds.Value("string"),
                "relavance_label": ds.features.ClassLabel(
                    names=["Bad", "Fair", "Good", "Excellent", "Perfect"]
                ),
            }
        elif self.config.name == "qam":
            features = {
                "question": ds.Value("string"),
                "answer": ds.Value("string"),
                "label": ds.features.ClassLabel(names=["False", "True"]),
            }
        elif self.config.name == "qg":
            features = {
                "answer_passage": ds.Value("string"),
                "question": ds.Value("string"),
            }
        elif self.config.name == "ntg":
            features = {
                "news_body": ds.Value("string"),
                "news_title": ds.Value("string"),
            }
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(features),
        )

    def _split_generators(self, mgr):
        all_data_folder = mgr.download_and_extract(_XGLUE_ALL_DATA)
        data_folder = os.path.join(all_data_folder, "xglue_full_dataset", self.config.data_dir)
        name = self.config.name
        languages = _LANGS[name]
        return (
            [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={
                        "data_file": os.path.join(data_folder, _PATHS[name]["train"]),
                        "split": "train",
                    },
                ),
            ]
            + [
                ds.SplitGenerator(
                    name=ds.Split(f"validation.{c}"),
                    gen_kw={
                        "data_file": os.path.join(data_folder, _PATHS[name]["dev"].format(c)),
                        "split": "dev",
                    },
                )
                for c in languages
            ]
            + [
                ds.SplitGenerator(
                    name=ds.Split(f"test.{x}"),
                    gen_kw={
                        "data_file": os.path.join(data_folder, _PATHS[name]["test"].format(x)),
                        "split": "test",
                    },
                )
                for x in languages
            ]
        )

    def _generate_examples(self, data_file, split=None):
        keys = list(self._info().features.keys())
        if self.config.name == "mlqa":
            with open(data_file, encoding="utf-8") as f:
                data = json.load(f)
            for examples in data["data"]:
                for example in examples["paragraphs"]:
                    context = example["context"]
                    for qa in example["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]
                        answers = qa["answers"]
                        answers_start = [answer["answer_start"] for answer in answers]
                        answers_text = [answer["text"] for answer in answers]
                        yield id_, {
                            "context": context,
                            "question": question,
                            "answers": {"answer_start": answers_start, "text": answers_text},
                        }
        elif self.config.name in ["ner", "pos"]:
            words = []
            result = []
            idx = -1
            with open(data_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip() == "":
                        if len(words) > 0:
                            y_kw = {keys[0]: words, keys[1]: result}
                            words = []
                            result = []
                            idx += 1
                            yield idx, y_kw
                    else:
                        splits = line.strip().split(" ")
                        words.append(splits[0])
                        result.append(splits[1])
        elif self.config.name in ["ntg", "qg"]:
            with open(data_file + ".src." + split, encoding="utf-8") as src_f, open(
                data_file + ".tgt." + split, encoding="utf-8"
            ) as tgt_f:
                for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                    yield idx, {keys[0]: src_line.strip(), keys[1]: tgt_line.strip()}
        else:
            _process_dict = {
                "paws-x": {"0": "different", "1": "same"},
                "xnli": {"contradictory": "contradiction"},
                "qam": {"0": "False", "1": "True"},
                "wpr": {"0": "Bad", "1": "Fair", "2": "Good", "3": "Excellent", "4": "Perfect"},
            }

            def _process(value):
                if self.config.name in _process_dict and value in _process_dict[self.config.name]:
                    return _process_dict[self.config.name][value]
                return value

            with open(data_file, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if data_file.split(".")[-1] == "tsv" and idx == 0:
                        continue
                    items = line.strip().split("\t")
                    yield idx, {
                        key: _process(value)
                        for key, value in zip(
                            keys, items[1:] if self.config.name == "paws-x" else items
                        )
                    }
