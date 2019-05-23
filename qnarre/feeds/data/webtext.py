
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
    url = 'https://storage.googleapis.com/gpt-2/output-dataset/v1/'
    suff = 'jsonl'
    for f, s in (
        ('webtext', ''),
        ('small-117M', ''),
        ('small-117M-k40', ''),
        ('medium-345M', ''),
        ('medium-345M-k40', ''),
        ('large-762M', ''),
        ('large-762M-k40', ''),
        ('xl-1542M', ''),
        ('xl-1542M-k40', ''),
    ):
        for k in ('train', 'valid', 'test'):
            sh.run('wget -q -c -N {}'.format(url + '.'.join((f, k, suff))),
                   # 'xz -qk -9 -T0 {}'.format(f + suff),
                   )
            # assert s == sh.sha256sum(f + suff)


def main(_):
    p = pth.Path.cwd() / flags.FLAGS.dir_data
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    download(Shell(p))


if __name__ == '__main__':
    flags.DEFINE_string(name='dir_data', default='.data/webtext', help='')
    from absl import app
    app.run(main)
