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
import tensorflow as tf

import datasets as qd
import utils as qu

np.random.seed(12345)

tf.config.experimental.set_visible_devices([], 'GPU')

ds_small = dict(
    dim_batch=5,
    dim_pool=10,
    max_val=1000,
    num_samples=20,
    num_shards=2,
)

ds_large = dict(
    dim_batch=1000,
    dim_pool=1024 * 1024,
    max_val=100000,
    num_samples=10000,
    num_shards=100,
)


def dump_ds(kind):
    ps = qu.Params(**(ds_small if kind == 'small' else ds_large))
    ss = [s for s in qd.dump(ps, f'/tmp/q/data/{kind}')]
    ds = qd.load(ps, shards=ss).map(qd.adapter)
    for i, _ in enumerate(ds):
        pass
    print(f'dumped {i + 1} batches of {ps.dim_batch} samples each')


if __name__ == '__main__':
    dump_ds('small')
    # dump_ds('large')
