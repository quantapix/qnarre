
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

import pickle

import tensorflow as tf
import subprocess as sub


def download_raw(base):
    p = base / 'cifar/raw'
    if not p.exists():
        p.mkdir(parents=True)
    u = 'https://www.cs.toronto.edu/~kriz/'
    for f in (
            'cifar-10-python.tar.gz',
            'cifar-10-binary.tar.gz',
            'cifar-100-python.tar.gz',
            'cifar-100-binary.tar.gz',
    ):
        cmd = 'wget --show-progress -q -c -N {}'.format(u + f)
        print(cmd)
        sub.run(cmd.split(), cwd=str(p))
        cmd = 'tar -xf raw/{}'.format(f)
        print(cmd)
        sub.run(cmd.split(), cwd=str(p.parent))


def convert(base):
    base /= 'cifar'
    s, d = base / 'cifar-10-batches-py', base / 'tf'
    if not d.exists():
        d.mkdir(parents=True)
    batches = 5
    ps = {
        'train': [s / 'data_batch_{}'.format(i + 1) for i in range(batches)],
        'test': [s / 'test_batch'],
    }
    for n, ss in ps.items():
        n = 'cifar-10-' + n + '.tfrecords'
        _to_tfrecords(ss, d / n)

    s = base / 'cifar-100-python'
    for k in ('fine', 'coarse'):
        ps = {
            'train-' + k: [s / 'train'],
            'test-' + k: [s / 'test'],
        }
        for n, ss in ps.items():
            n = 'cifar-100-' + n + '.tfrecords'
            _to_tfrecords(ss, d / n, k + '_labels')


def _to_tfrecords(srcs, dst, labels='labels'):
    def _int64(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    def _bytes(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    with tf.python_io.TFRecordWriter(str(dst)) as w:
        for s in srcs:
            with open(s, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
            d = {k.decode(): v for k, v in d.items()}
            x, y = d['data'], d[labels]
            for i in range(len(y)):
                e = tf.train.Example(
                    features=tf.train.Features(feature={
                        'x': _bytes(x[i].tobytes()),
                        'y': _int64(y[i])
                    }))
                w.write(e.SerializeToString())


if __name__ == "__main__":
    import pathlib as pth
    b = pth.Path.home() / 'cache/qnets'
    download_raw(b)
    convert(b)
