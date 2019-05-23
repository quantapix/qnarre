
# Copyright 2019 Quantapix Authors. All Rights Reserved.
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

import pathlib as pth

from absl import flags

from qnarre.feeds.data.shell import Shell


def download(sh):
    url = 'http://www.statmt.org/lm-benchmark/'
    suff = '.tar.gz'
    for f, s in (
        ('1-billion-word-language-modeling-benchmark-r13output',
         '01ba60381110baf7f189dfd2b8374de371e8c9a340835793f190bdae9e90a34e'),
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            # 'xz -qk -9 -T0 {}'.format(f + suff),
        )
        assert s == sh.sha256sum(f + suff)

    url = 'https://github.com/rafaljozefowicz/lm/raw/master/'
    suff = '.txt'
    for f, s in (
        ('1b_word_vocab',
         'c758fee313a5c952995f391a659fe31cde50f2b11f5fae6c0e75a0a6e79fa34d'),
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            # 'xz -qk -9 -T0 {}'.format(f + suff),
        )
        assert s == sh.sha256sum(f + suff)


def main(_):
    p = pth.Path.cwd() / flags.FLAGS.dir_data
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    download(Shell(p))


if __name__ == '__main__':
    flags.DEFINE_string(name='dir_data', default='.data/1b_word', help='')
    from absl import app
    app.run(main)
