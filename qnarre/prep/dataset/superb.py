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

import csv
import glob
import os
from dataclasses import dataclass

import datasets as ds
from datasets.tasks import AutomaticSpeechRecognition


class SuperbConfig(ds.BuilderConfig):
    def __init__(
        self,
        features,
        url,
        data_url=None,
        supervised_keys=None,
        task_templates=None,
        **kw,
    ):
        super().__init__(version=ds.Version("1.9.0", ""), **kw)
        self.features = features
        self.data_url = data_url
        self.url = url
        self.supervised_keys = supervised_keys
        self.task_templates = task_templates


class Superb(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SuperbConfig(
            name="asr",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "text": ds.Value("string"),
                    "speaker_id": ds.Value("int64"),
                    "chapter_id": ds.Value("int64"),
                    "id": ds.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            data_url="http://www.openslr.org/resources/12/",
            task_templates=[
                AutomaticSpeechRecognition(
                    audio_file_path_column="file", transcription_column="text"
                )
            ],
        ),
        SuperbConfig(
            name="ks",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "label": ds.ClassLabel(
                        names=[
                            "yes",
                            "no",
                            "up",
                            "down",
                            "left",
                            "right",
                            "on",
                            "off",
                            "stop",
                            "go",
                            "_silence_",
                            "_unknown_",
                        ]
                    ),
                }
            ),
            supervised_keys=("file", "label"),
            data_url="http://download.tensorflow.org/data/{filename}",
        ),
        SuperbConfig(
            name="ic",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "speaker_id": ds.Value("string"),
                    "text": ds.Value("string"),
                    "action": ds.ClassLabel(
                        names=[
                            "activate",
                            "bring",
                            "change language",
                            "deactivate",
                            "decrease",
                            "increase",
                        ]
                    ),
                    "object": ds.ClassLabel(
                        names=[
                            "Chinese",
                            "English",
                            "German",
                            "Korean",
                            "heat",
                            "juice",
                            "lamp",
                            "lights",
                            "music",
                            "newspaper",
                            "none",
                            "shoes",
                            "socks",
                            "volume",
                        ]
                    ),
                    "location": ds.ClassLabel(names=["bedroom", "kitchen", "none", "washroom"]),
                }
            ),
            supervised_keys=None,
            data_url="http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz",
        ),
        SuperbConfig(
            name="si",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "label": ds.ClassLabel(names=[f"id{i + 10001}" for i in range(1251)]),
                }
            ),
            supervised_keys=("file", "label"),
        ),
        SuperbConfig(
            name="sd",
            features=ds.Features(
                {
                    "record_id": ds.Value("string"),
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "start": ds.Value("int64"),
                    "end": ds.Value("int64"),
                    "speakers": [
                        {
                            "speaker_id": ds.Value("string"),
                            "start": ds.Value("int64"),
                            "end": ds.Value("int64"),
                        }
                    ],
                }
            ),
            supervised_keys=None,
            data_url="https://huggingface.co/datasets/superb/superb-data/resolve/main/sd/{split}/{filename}",
        ),
        SuperbConfig(
            name="er",
            features=ds.Features(
                {
                    "file": ds.Value("string"),
                    "audio": ds.Audio(sampling_rate=16_000),
                    "label": ds.ClassLabel(names=["neu", "hap", "ang", "sad"]),
                }
            ),
            supervised_keys=("file", "label"),
        ),
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            features=self.config.features,
            supervised_keys=self.config.supervised_keys,
            homepage="",
            citation="",
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "asr":
            _DL_URLS = {
                "dev": self.config.data_url + "dev-clean.tar.gz",
                "test": self.config.data_url + "test-clean.tar.gz",
                "train": self.config.data_url + "train-clean-100.tar.gz",
            }
            archive_path = dl_manager.download_and_extract(_DL_URLS)
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN, gen_kw={"archive_path": archive_path["train"]}
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION, gen_kw={"archive_path": archive_path["dev"]}
                ),
                ds.SplitGenerator(
                    name=ds.Split.TEST, gen_kw={"archive_path": archive_path["test"]}
                ),
            ]
        elif self.config.name == "ks":
            _DL_URLS = {
                "train_val_test": self.config.data_url.format(
                    filename="speech_commands_v0.01.tar.gz"
                ),
                "test": self.config.data_url.format(
                    filename="speech_commands_test_set_v0.01.tar.gz"
                ),
            }
            archive_path = dl_manager.download_and_extract(_DL_URLS)
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"archive_path": archive_path["train_val_test"], "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"archive_path": archive_path["train_val_test"], "split": "val"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"archive_path": archive_path["test"], "split": "test"},
                ),
            ]
        elif self.config.name == "ic":
            archive_path = dl_manager.download_and_extract(self.config.data_url)
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"archive_path": archive_path, "split": "train"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"archive_path": archive_path, "split": "valid"},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TEST,
                    gen_kw={"archive_path": archive_path, "split": "test"},
                ),
            ]
        elif self.config.name == "si":
            manual_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kw={"archive_path": manual_dir, "split": 1},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kw={"archive_path": manual_dir, "split": 2},
                ),
                ds.SplitGenerator(
                    name=ds.Split.TEST, gen_kw={"archive_path": manual_dir, "split": 3}
                ),
            ]
        elif self.config.name == "sd":
            splits = ["train", "dev", "test"]
            _DL_URLS = {
                split: {
                    filename: self.config.data_url.format(split=split, filename=filename)
                    for filename in ["reco2dur", "segments", "utt2spk", "wav.zip"]
                }
                for split in splits
            }
            archive_path = dl_manager.download_and_extract(_DL_URLS)
            return [
                ds.SplitGenerator(
                    name=ds.NamedSplit(split),
                    gen_kw={"archive_path": archive_path[split], "split": split},
                )
                for split in splits
            ]
        elif self.config.name == "er":
            manual_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
            return [
                ds.SplitGenerator(
                    name=f"session{i}",
                    gen_kw={"archive_path": manual_dir, "split": i},
                )
                for i in range(1, 6)
            ]

    def _generate_examples(self, archive_path, split=None):
        if self.config.name == "asr":
            transcripts_glob = os.path.join(archive_path, "LibriSpeech", "*", "*", "*", "*.txt")
            key = 0
            for transcript_path in sorted(glob.glob(transcripts_glob)):
                transcript_dir_path = os.path.dirname(transcript_path)
                with open(transcript_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        id_, transcript = line.split(" ", 1)
                        audio_file = f"{id_}.flac"
                        speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                        audio_path = os.path.join(transcript_dir_path, audio_file)
                        yield key, {
                            "id": id_,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "file": audio_path,
                            "audio": audio_path,
                            "text": transcript,
                        }
                        key += 1
        elif self.config.name == "ks":
            words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
            splits = _split_ks_files(archive_path, split)
            for key, audio_file in enumerate(sorted(splits[split])):
                base_dir, file_name = os.path.split(audio_file)
                _, word = os.path.split(base_dir)
                if word in words:
                    label = word
                elif word == "_silence_" or word == "_background_noise_":
                    label = "_silence_"
                else:
                    label = "_unknown_"
                yield key, {"file": audio_file, "audio": audio_file, "label": label}
        elif self.config.name == "ic":
            root_path = os.path.join(archive_path, "fluent_speech_commands_dataset")
            csv_path = os.path.join(root_path, "data", f"{split}_data.csv")
            with open(csv_path, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",", skipinitialspace=True)
                next(csv_reader)
                for row in csv_reader:
                    key, file_path, speaker_id, text, action, object_, location = row
                    audio_path = os.path.join(root_path, file_path)
                    yield key, {
                        "file": audio_path,
                        "audio": audio_path,
                        "speaker_id": speaker_id,
                        "text": text,
                        "action": action,
                        "object": object_,
                        "location": location,
                    }
        elif self.config.name == "si":
            wav_path = os.path.join(archive_path, "wav")
            splits_path = os.path.join(archive_path, "veri_test_class.txt")
            with open(splits_path, "r", encoding="utf-8") as f:
                for key, line in enumerate(f):
                    split_id, file_path = line.strip().split(" ")
                    if int(split_id) != split:
                        continue
                    speaker_id = file_path.split("/")[0]
                    audio_path = os.path.join(wav_path, file_path)
                    yield key, {
                        "file": audio_path,
                        "audio": audio_path,
                        "label": speaker_id,
                    }
        elif self.config.name == "sd":
            data = SdData(archive_path)
            args = SdArgs()
            chunk_indices = _generate_chunk_indices(data, args, split=split)
            if split != "test":
                for key, (rec, st, ed) in enumerate(chunk_indices):
                    speakers = _get_speakers(rec, data, args)
                    yield key, {
                        "record_id": rec,
                        "file": data.wavs[rec],
                        "audio": data.wavs[rec],
                        "start": st,
                        "end": ed,
                        "speakers": speakers,
                    }
            else:
                key = 0
                for rec in chunk_indices:
                    for rec, st, ed in chunk_indices[rec]:
                        speakers = _get_speakers(rec, data, args)
                        yield key, {
                            "record_id": rec,
                            "file": data.wavs[rec],
                            "audio": data.wavs[rec],
                            "start": st,
                            "end": ed,
                            "speakers": speakers,
                        }
                        key += 1
        elif self.config.name == "er":
            root_path = os.path.join(archive_path, f"Session{split}")
            wav_path = os.path.join(root_path, "sentences", "wav")
            labels_path = os.path.join(root_path, "dialog", "EmoEvaluation", "*.txt")
            emotions = ["neu", "hap", "ang", "sad", "exc"]
            key = 0
            for labels_file in sorted(glob.glob(labels_path)):
                with open(labels_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line[0] != "[":
                            continue
                        _, filename, emo, _ = line.split("\t")
                        if emo not in emotions:
                            continue
                        wav_subdir = filename.rsplit("_", 1)[0]
                        filename = f"{filename}.wav"
                        audio_path = os.path.join(wav_path, wav_subdir, filename)
                        yield key, {
                            "file": audio_path,
                            "audio": audio_path,
                            "label": emo.replace("exc", "hap"),
                        }
                        key += 1


class SdData:
    def __init__(self, data_dir):
        self.segments = self._load_segments_rechash(data_dir["segments"])
        self.utt2spk = self._load_utt2spk(data_dir["utt2spk"])
        self.wavs = self._load_wav_zip(data_dir["wav.zip"])
        self.reco2dur = self._load_reco2dur(data_dir["reco2dur"])

    def _load_segments_rechash(self, segments_file):
        ret = {}
        if not os.path.exists(segments_file):
            return None
        with open(segments_file, encoding="utf-8") as f:
            for line in f:
                utt, rec, st, et = line.strip().split()
                if rec not in ret:
                    ret[rec] = []
                ret[rec].append({"utt": utt, "st": float(st), "et": float(et)})
        return ret

    def _load_wav_zip(self, wav_zip):
        wav_dir = os.path.join(wav_zip, "wav")
        return {
            os.path.splitext(filename)[0]: os.path.join(wav_dir, filename)
            for filename in sorted(os.listdir(wav_dir))
        }

    def _load_utt2spk(self, utt2spk_file):
        with open(utt2spk_file, encoding="utf-8") as f:
            lines = [line.strip().split(None, 1) for line in f]
        return {x[0]: x[1] for x in lines}

    def _load_reco2dur(self, reco2dur_file):
        if not os.path.exists(reco2dur_file):
            return None
        with open(reco2dur_file, encoding="utf-8") as f:
            lines = [line.strip().split(None, 1) for line in f]
        return {x[0]: float(x[1]) for x in lines}


@dataclass
class SdArgs:
    chunk_size = 2000
    frame_shift = 160
    subsampling = 1
    label_delay = 0
    num_speakers = 2
    rate = 16000
    use_last_samples = True


def _generate_chunk_indices(data, args, split=None):
    chunk_indices = [] if split != "test" else {}
    for rec in data.wavs:
        data_len = int(data.reco2dur[rec] * args.rate / args.frame_shift)
        data_len = int(data_len / args.subsampling)
        if split == "test":
            chunk_indices[rec] = []
        if split != "test":
            for st, ed in _gen_frame_indices(
                data_len,
                args.chunk_size,
                args.chunk_size,
                args.use_last_samples,
                label_delay=args.label_delay,
                subsampling=args.subsampling,
            ):
                chunk_indices.append((rec, st * args.subsampling, ed * args.subsampling))
        else:
            for st, ed in _gen_chunk_indices(data_len, args.chunk_size):
                chunk_indices[rec].append((rec, st * args.subsampling, ed * args.subsampling))
    return chunk_indices


def _count_frames(data_len, size, step):
    return int((data_len - size + step) / step)


def _gen_frame_indices(
    data_length, size=2000, step=2000, use_last_samples=False, label_delay=0, subsampling=1
):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def _get_speakers(rec, data, args):
    return [
        {
            "speaker_id": data.utt2spk[segment["utt"]],
            "start": round(segment["st"] * args.rate / args.frame_shift),
            "end": round(segment["et"] * args.rate / args.frame_shift),
        }
        for segment in data.segments[rec]
    ]


def _split_ks_files(archive_path, split):
    audio_path = os.path.join(archive_path, "**", "*.wav")
    audio_paths = glob.glob(audio_path)
    if split == "test":
        return {"test": audio_paths}
    val_list_file = os.path.join(archive_path, "validation_list.txt")
    test_list_file = os.path.join(archive_path, "testing_list.txt")
    with open(val_list_file, encoding="utf-8") as f:
        val_paths = f.read().strip().splitlines()
        val_paths = [os.path.join(archive_path, p) for p in val_paths]
    with open(test_list_file, encoding="utf-8") as f:
        test_paths = f.read().strip().splitlines()
        test_paths = [os.path.join(archive_path, p) for p in test_paths]
    train_paths = list(set(audio_paths) - set(val_paths) - set(test_paths))
    return {"train": train_paths, "val": val_paths}
