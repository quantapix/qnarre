
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
    url = 'http://curtis.ml.cmu.edu/datasets/pretrained_xl'
    for p in ('enwiki8', 'text8', 'wt103', ('lm1b', '1191000')):
        p, v = p if isinstance(p, tuple) else (p, '0')
        for d, f in (
            ('data', 'cache.pkl'),
            ('data', 'corpus-info.json'),
            ('model', 'checkpoint'),
            ('model', 'model.ckpt-#.data-00000-of-00001'),
            ('model', 'model.ckpt-#.index'),
            ('model', 'model.ckpt-#.meta'),
        ):
            f = f.replace('#', v)
            sh.run(
                f'mkdir -p {p}',
                f'cd {p}',
                'wget -q -c -N {}'.format('/'.join((url, 'tf_' + p, d, f))),
                # 'xz -qk -9 -T0 {}'.format(f + suff),
            )


def main(_):
    p = pth.Path.cwd() / flags.FLAGS.dir_data
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    download(Shell(p))


if __name__ == '__main__':
    flags.DEFINE_string(name='dir_data', default='.model/trafo_xl', help='')
    from absl import app
    app.run(main)
