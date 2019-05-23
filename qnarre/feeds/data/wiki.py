
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
# https://arxiv.org/pdf/1801.10198.pdf
# https://arxiv.org/pdf/1701.06548.pdf

import pathlib as pth

from absl import flags

from qnarre.feeds.data.shell import Shell


def download(sh):
    files = ('enwiki-latest-pages-articles-multistream.xml', )
    url = 'ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/latest/'
    suff = '.bz2'
    for f in files:
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            'mv {} wikidump.xml{}'.format(f + suff, suff),
            'bunzip2 -qk wikidump.xml{}'.format(suff),
        )


def define_download_flags():
    flags.DEFINE_string(name='dir_data', default='.data/wiki', help='Data dir')


def main(_):
    path = pth.Path.cwd() / flags.FLAGS.dir_data
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    # download(Shell(path))


if __name__ == '__main__':
    define_download_flags()
    from absl import app
    app.run(main)
"""# Or generate your own using https://github.com/attardi/wikiextractor
aws s3 cp s3://yaroslavvb2/data/wikiextracted.tar .
tar -xf wikiextracted.tar
# Flatten all the files.
find wikiextracted/ -iname 'wiki*' -type f -exec sh -c 'jq -r .text {} > {}.txt' \;
"""
