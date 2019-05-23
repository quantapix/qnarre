
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
# https://arxiv.org/pdf/1806.03822.pdf
# https://arxiv.org/pdf/1606.05250.pdf

import pathlib as pth

from absl import flags

from qnarre.feeds.data.shell import Shell


def download(sh):
    url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
    suff = '.json'
    for f, s in (
        ('train-v1.1',
         '3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955'),
        ('dev-v1.1',
         '95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9'),
        ('train-v2.0',
         '68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002'),
        ('dev-v2.0',
         '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'),
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + f + suff),
            'xz -qk -9 -T0 {}'.format(f + suff),
        )
        assert s == sh.sha256sum(f + suff)

    url = 'https://worksheets.codalab.org/rest/bundles/'
    for b, f, s in (
        ('0xbcd57bee090b421c982906709c8c27e1/contents/blob/',
         'evaluate-v1.1.py',
         'f5a673dbbd173e29e9ea38f1b2091d883583b77b3a4c17144b223fb0f2f9bd09'),
        ('0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/',
         'evaluate-v2.0.py',
         '710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4'),
    ):
        sh.run(
            'wget -q -c -N {}'.format(url + b),
            'mv index.html {}'.format(f),
        )
        assert s == sh.sha256sum(f)


def main(_):
    p = pth.Path.cwd() / flags.FLAGS.dir_data
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    download(Shell(p))


if __name__ == '__main__':
    flags.DEFINE_string(name='dir_data', default='.data/squad', help='')
    from absl import app
    app.run(main)
