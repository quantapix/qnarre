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

import os

import pandas as pd

import datasets as ds
from datasets.tasks import AutomaticSpeechRecognition

_URL = "https://data.deepai.org/timit.zip"


class TimitASR(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ds.BuilderConfig(name="clean", version=ds.Version("2.0.1"))]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "text": ds.Value("string"),
                    "phonetic_detail": ds.Sequence(
                        {
                            "start": ds.Value("int64"),
                            "stop": ds.Value("int64"),
                            "utterance": ds.Value("string"),
                        }
                    ),
                    "word_detail": ds.Sequence(
                        {
                            "start": ds.Value("int64"),
                            "stop": ds.Value("int64"),
                            "utterance": ds.Value("string"),
                        }
                    ),
                    "dialect_region": ds.Value("string"),
                    "sentence_type": ds.Value("string"),
                    "speaker_id": ds.Value("string"),
                    "id": ds.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            task_templates=[
                AutomaticSpeechRecognition(
                    audio_file_path_column="file", transcription_column="text"
                )
            ],
        )

    def _split_generators(self, mgr):
        p = mgr.download_and_extract(_URL)
        train = os.path.join(p, "train_data.csv")
        test = os.path.join(p, "test_data.csv")
        return [
            ds.SplitGenerator(name=ds.Split.TRAIN, gen_kw={"data_info_csv": train}),
            ds.SplitGenerator(name=ds.Split.TEST, gen_kw={"data_info_csv": test}),
        ]

    def _generate_examples(self, data_info_csv):
        data_path = os.path.join(os.path.dirname(data_info_csv).strip(), "data")
        data_info = pd.read_csv(open(data_info_csv, encoding="utf8"))
        data_info.dropna(subset=["path_from_data_dir"], inplace=True)
        data_info = data_info.loc[(data_info["is_audio"]) & (~data_info["is_converted_audio"])]
        for i in range(data_info.shape[0]):
            audio = data_info.iloc[i]
            wav = os.path.join(data_path, *(audio["path_from_data_dir"].split("/")))
            with open(wav.replace(".WAV", ".TXT"), encoding="utf-8") as op:
                transcript = " ".join(op.readlines()[0].split()[2:])
            with open(wav.replace(".WAV", ".PHN"), encoding="utf-8") as op:
                phonemes = [
                    {
                        "start": l.split(" ")[0],
                        "stop": l.split(" ")[1],
                        "utterance": " ".join(l.split(" ")[2:]).strip(),
                    }
                    for l in op.readlines()
                ]
            with open(wav.replace(".WAV", ".WRD"), encoding="utf-8") as op:
                words = [
                    {
                        "start": l.split(" ")[0],
                        "stop": l.split(" ")[1],
                        "utterance": " ".join(l.split(" ")[2:]).strip(),
                    }
                    for l in op.readlines()
                ]
            y = {
                "file": wav,
                "audio": wav,
                "text": transcript,
                "phonetic_detail": phonemes,
                "word_detail": words,
                "dialect_region": audio["dialect_region"],
                "sentence_type": audio["filename"][0:2],
                "speaker_id": audio["speaker_id"],
                "id": audio["filename"].replace(".WAV", ""),
            }
            yield i, y
