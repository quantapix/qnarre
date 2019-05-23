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

# tf.serialize_tensor <--> tf.parse_tensor


def bytes_feat(v):
    return many_bytes_feat([v])


def many_bytes_feat(vs):
    assert isinstance(vs, list)

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


def dump(path, examples):
    path.parent.mkdir()
    with tf.TFRecordWriter(str(path)) as w:
        for e in examples():
            w.write(e)
