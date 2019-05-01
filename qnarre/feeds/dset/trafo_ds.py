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

import tensorflow as T


def dataset(params, _):
    PS = params
    PS.update(PAD=0, UNK=1, BEG=2, END=3, vocab_size=20, tgt_len=PS.ctx_len)
    t, sh = T.int32, T.TensorShape((PS.ctx_len, ))
    return T.data.Dataset.from_generator(
        lambda: _generator(PS),
        ((t, t, t), t),
        ((sh, sh, sh), sh),
    )


def _generator(PS):
    sl = PS.ctx_len
    for _ in range(10000):
        n = randint(1, sl - 2)
        c = randint(0, 9) + 10
        s = [PS.BEG] + [c] * n + [PS.END] + [PS.PAD] * (sl - n - 2)
        yield (s, [0] * sl, [PS.UNK] * sl), s
