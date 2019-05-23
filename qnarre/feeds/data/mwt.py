
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
# http://www.statmt.org/wmt18/translation-task.html

import pathlib as pth

from absl import flags

from official.utils.flags import core as fu

from ..shell import Shell


def download(sh):
    files = (
        ('http://data.statmt.org/wmt18/translation-task/',
         'training-parallel-nc-v13', 'train'),
        ('http://www.statmt.org/wmt13/', 'training-parallel-commoncrawl',
         'train2'),
        ('http://www.statmt.org/wmt13/', 'training-parallel-europarl-v7',
         'train3'),
        ('http://data.statmt.org/wmt18/translation-task/', 'dev', 'eval'),
    )
    suff = '.tgz'
    for u, s, d in files:
        sh.run(
            # 'wget --show-progress -q -c -N {}'.format(u + s + suff),
            'wget -q -c -N {}'.format(u + s + suff),
            'tar xf {}'.format(s + suff),
            'mv {} {}'.format(s, d),
        )


def define_download_flags():
    fu.define_base()
    flags.adopt_module_key_flags(fu)
    fu.set_defaults(dir_data='.cache/transformer/data')


def main(_):
    path = pth.Path.cwd() / flags.FLAGS.dir_data
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    download(Shell(path))


if __name__ == '__main__':
    define_download_flags()
    from absl import app
    app.run(main)
