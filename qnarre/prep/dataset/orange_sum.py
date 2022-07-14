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


_URL = {
    "abstract": "https://raw.githubusercontent.com/Tixierae/OrangeSum/main/data/docs/splits/abstract.tgz",
    "title": "https://raw.githubusercontent.com/Tixierae/OrangeSum/main/data/docs/splits/title.tgz",
}

_DOC = "text"
_SUM = "summary"


class OrangeSum(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(name="abstract", version=ds.Version("1.1.0")),
        ds.BuilderConfig(name="title", version=ds.Version("1.1.0")),
    ]

    def _info(self):
        return ds.DatasetInfo(
            description="",
            citation="",
            homepage="",
            license="",
            features=ds.Features(
                {
                    _DOC: ds.Value("string"),
                    _SUM: ds.Value("string"),
                }
            ),
            supervised_keys=(_DOC, _SUM),
        )

    def _split_generators(self, mgr):
        x = mgr.download(_URL[self.config.name])
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kw={
                    "source_files": mgr.iter_archive(x),
                    "target_files": mgr.iter_archive(x),
                    "split": "train",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kw={
                    "source_files": mgr.iter_archive(x),
                    "target_files": mgr.iter_archive(x),
                    "split": "test",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kw={
                    "source_files": mgr.iter_archive(x),
                    "target_files": mgr.iter_archive(x),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, src, tgt, split):
        sp = f"{self.config.name}/{split}.source"
        tp = f"{self.config.name}/{split}.target"
        for x, fx in src:
            if x == sp:
                for y, fy in tgt:
                    if y == tp:
                        for i, (d, s) in enumerate(zip(fx, fy)):
                            yield i, {_DOC: d.decode("utf-8"), _SUM: s.decode("utf-8")}
                        break
                break
