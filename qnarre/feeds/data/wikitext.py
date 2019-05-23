
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
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    suff = '.zip'
    for f, s in (
        ('wikitext-103-v1',
         '242ba0f20b329cfdf1ccc61e9e9e5b59becf189db7f7a81cd2a0e2fc31539590'),
        ('wikitext-103-raw-v1',
         '91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794'),
        ('wikitext-2-v1',
         '92675f1d63015c1c8b51f1656a52d5bdbc33aafa60cc47a218a66e7ee817488c'),
        ('wikitext-2-raw-v1',
         'ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11')
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            'xz -qk -9 -T0 {}'.format(f + suff),
        )
        assert s == sh.sha256sum(f + suff)


def main(_):
    p = pth.Path.cwd() / flags.FLAGS.dir_data
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    download(Shell(p))


if __name__ == '__main__':
    flags.DEFINE_string(name='dir_data', default='.data/wikitext', help='')
    from absl import app
    app.run(main)
