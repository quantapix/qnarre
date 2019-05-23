
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

from qfeeds.data.shell import Shell


def download(sh):
    url = 'http://nlp.stanford.edu/data/'
    suff = '.zip'
    for f, s in (
        ('glove.840B.300d',
         'c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f'),
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            'unzip -quo {}'.format(f + suff),
        )
        assert s == sh.sha256sum(f + suff)
    url = 'https://raw.githubusercontent.com/minimaxir/char-embeddings/master/'
    suff = '.txt'
    for f, s in (
        ('glove.840B.300d-char',
         '4372b6d8b1eb7e7e076021732ac6965b9b33ab84cc44cd0ca4db9cf967abaa43'),
    ):
        sh.run('wget -q -c -N {}'.format(url + f + suff), )
        assert s == sh.sha256sum(f + suff)


def define_download_flags():
    flags.DEFINE_string(
        name='dir_data', default='.data/glove', help='Data dir')


def main(_):
    path = pth.Path.cwd() / flags.FLAGS.dir_data
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    download(Shell(path))


if __name__ == '__main__':
    define_download_flags()
    from absl import app
    app.run(main)
