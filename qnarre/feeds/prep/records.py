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

import numpy as np

from qnarre.neura import tf


def bytes_feat(v):
    return bytes_list_feat([v])


def bytes_list_feat(vs):
    assert isinstance(vs, list) or isinstance(vs, np.ndarray)

    def unpack():
        for v in vs:
            yield v.numpy() if isinstance(v, type(tf.constant(0))) else v

    return tf.Feature(bytes_list=tf.BytesList(value=list(unpack())))


def one_float_feat(v):
    return floats_feat([v])


def floats_feat(vs):
    assert isinstance(vs, list) or isinstance(vs, np.ndarray)
    return tf.Feature(float_list=tf.FloatList(value=vs))


def one_int_feat(v):
    return ints_feat([v])


def ints_feat(vs):
    assert isinstance(vs, list) or isinstance(vs, np.ndarray)
    return tf.Feature(int64_list=tf.Int64List(value=vs))


def example(feat):
    e = tf.Example(features=tf.Features(feature=feat))
    return e.SerializeToString()


def dataset(path):
    p = path.with_suffix('.tfrecords')
    return tf.TFRecordDataset(str(p))


@tf.function
def load(path, feat):
    ds = dataset(path).take(1)
    for d in ds.map(lambda x: tf.parse_single_example(x, feat)):
        return tf.to_dense(d['toks']).numpy()


def dump(path, records):
    p = path.with_suffix('.tfrecords')
    p.parent.mkdir()
    with tf.TFRecordWriter(str(p)) as w:
        for r in records():
            w.write(r)
