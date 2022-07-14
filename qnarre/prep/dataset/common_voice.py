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
from datasets.tasks import AutomaticSpeechRecognition


_DATA_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{}.tar.gz"

_LANGS = {
    "de": {
        "Language": "German",
        "Date": "2020-12-11",
        "Size": "22 GB",
        "Version": "de_836h_2020-12-11",
        "Validated_Hr_Total": 777,
        "Overall_Hr_Total": 836,
        "Number_Of_Voice": 12659,
    },
    "en": {
        "Language": "English",
        "Date": "2020-12-11",
        "Size": "56 GB",
        "Version": "en_2181h_2020-12-11",
        "Validated_Hr_Total": 1686,
        "Overall_Hr_Total": 2181,
        "Number_Of_Voice": 66173,
    },
    "ro": {
        "Language": "Romanian",
        "Date": "2020-12-11",
        "Size": "250 MB",
        "Version": "ro_9h_2020-12-11",
        "Validated_Hr_Total": 6,
        "Overall_Hr_Total": 9,
        "Number_Of_Voice": 130,
    },
}


class Config(ds.BuilderConfig):
    def __init__(self, name, sub_version, **kw):
        self.sub_version = sub_version
        self.language = kw.pop("language", None)
        self.date_of_snapshot = kw.pop("date", None)
        self.size = kw.pop("size", None)
        self.validated_hr_total = kw.pop("val_hrs", None)
        self.total_hr_total = kw.pop("total_hrs", None)
        self.num_of_voice = kw.pop("num_of_voice", None)
        super(Config, self).__init__(name=name, version=ds.Version("6.1.0", ""), **kw)


class CommonVoice(ds.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [
        Config(
            name=k,
            language=_LANGS[k]["Language"],
            sub_version=_LANGS[k]["Version"],
            date=_LANGS[k]["Date"],
            size=_LANGS[k]["Size"],
            val_hrs=_LANGS[k]["Validated_Hr_Total"],
            total_hrs=_LANGS[k]["Overall_Hr_Total"],
            num_of_voice=_LANGS[k]["Number_Of_Voice"],
        )
        for k in _LANGS.keys()
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    "client_id": ds.Value("string"),
                    "path": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=48_000),
                    "sentence": ds.Value("string"),
                    "up_votes": ds.Value("int64"),
                    "down_votes": ds.Value("int64"),
                    "age": ds.Value("string"),
                    "gender": ds.Value("string"),
                    "accent": ds.Value("string"),
                    "locale": ds.Value("string"),
                    "segment": ds.Value("string"),
                }
            ),
            task_templates=[
                AutomaticSpeechRecognition(
                    audio_file_path_column="path", transcription_column="sentence"
                )
            ],
        )

    def _split_generators(self, mgr):
        fs = mgr.download(_DATA_URL.format(self.config.name))
        data = "/".join(["cv-corpus-6.1-2020-12-11", self.config.name])
        clips = "/".join([data, "clips"])
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "train.tsv"]),
                    "path_to_clips": clips,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "test.tsv"]),
                    "path_to_clips": clips,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "dev.tsv"]),
                    "path_to_clips": clips,
                },
            ),
            ds.SplitGenerator(
                name="other",
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "other.tsv"]),
                    "path_to_clips": clips,
                },
            ),
            ds.SplitGenerator(
                name="validated",
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "validated.tsv"]),
                    "path_to_clips": clips,
                },
            ),
            ds.SplitGenerator(
                name="invalidated",
                gen_kw={
                    "files": mgr.iter_archive(fs),
                    "filepath": "/".join([data, "invalidated.tsv"]),
                    "path_to_clips": clips,
                },
            ),
        ]

    def _generate_examples(self, fs, filepath, clips):
        data_fields = list(self._info().features.keys())
        data_fields.remove("audio")
        i = data_fields.index("path")
        all_field_values = {}
        meta = False
        for p, f in fs:
            if p == filepath:
                meta = True
                lines = f.readlines()
                headline = lines[0].decode("utf-8")
                column_names = headline.strip().split("\t")
                assert (
                    column_names == data_fields
                ), f"The file should have {data_fields} as column names, but has {column_names}"
                for line in lines[1:]:
                    field_values = line.decode("utf-8").strip().split("\t")
                    audio_path = "/".join([clips, field_values[i]])
                    all_field_values[audio_path] = field_values
            elif p.startswith(clips):
                assert meta
                if not all_field_values:
                    break
                if p in all_field_values:
                    field_values = all_field_values[p]
                    if len(field_values) < len(data_fields):
                        field_values += (len(data_fields) - len(field_values)) * ["''"]
                    y = {k: v for k, v in zip(data_fields, field_values)}
                    y["audio"] = {"path": p, "bytes": f.read()}
                    yield p, y
