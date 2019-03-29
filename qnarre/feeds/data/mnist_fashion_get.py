#!/usr/bin/env python
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

from official.utils.flags import core as fu

from ..shell import Shell


def download(sh):
    files = (('train-images-idx3-ubyte',
              'train_images'), ('train-labels-idx1-ubyte', 'train_labels'),
             ('t10k-images-idx3-ubyte',
              'test_images'), ('t10k-labels-idx1-ubyte', 'test_labels'))
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    suff = '.gz'
    for s, d in files:
        sh.run(
            'wget -q -c -N {}'.format(url + s + suff),
            'gunzip -q -k {}'.format(s + suff),
            'mv {} {}'.format(s, d),
        )


def define_download_flags():
    fu.define_base()
    flags.adopt_module_key_flags(fu)
    fu.set_defaults(data_dir='.data/mnist-fashion')


def main(_):
    path = pth.Path.cwd() / flags.FLAGS.data_dir
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    download(Shell(path))


if __name__ == '__main__':
    define_download_flags()
    from absl import app
    app.run(main)
