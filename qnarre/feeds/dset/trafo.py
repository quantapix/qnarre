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

from random import randint

from qnarre.neura import tf


def dset(ps, _):
    ps.update(PAD=0, UNK=1, BEG=2, END=3, num_toks=20)
    ps.update(len_tgt=ps.len_src)
    t, sh = tf.int32, tf.TensorShape((ps.len_src, ))
    return tf.Dataset.from_generator(
        lambda: _generator(ps),
        ((t, ) * 4, t),
        ((sh, ) * 4, sh),
    )


def _generator(ps):
    sl = ps.len_src
    for _ in range(10000):
        n = randint(1, sl - 2)
        c = randint(0, 9) + 10
        s = [ps.BEG] + [c] * n + [ps.END] + [ps.PAD] * (sl - n - 2)
        h = [ps.BEG] + [ps.UNK] * (sl - 1)
        t = s[:n + 2] + h[n + 2:]
        yield (s, [0] * sl, h, t), t
