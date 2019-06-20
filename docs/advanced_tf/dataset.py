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
# !pip install -U tf-nightly-2.0-preview
# !pip install tensorflow==2.0.0-beta0
# export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

import numpy as np
import tensorflow as tf

td = tf.data
# ks = tf.keras
# kl = ks.layers

# complex input pipelines from simple, reusable pieces
# a pipeline starts with a "source" and chains "transformations" to it

# example - fascinating https://arxiv.org/pdf/1812.02825.pdf
# num_items of "x=-12,y=24:y+x:12" w/ "defs", "op" and "res"
# names: x, y, ops: +, -, *, values: [-max_val, max_val]

vocab = ('x', 'y')
vocab += ('+', '-', '*', '=', ',', ':')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# pipeline is parametric

params = dict(
    max_val=10,
    num_items=4,
)


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def native_source(ps):
    m, n = ps.max_val, ps.num_items
    # x, y values in defs
    vals = np.random.randint(low=-m, high=m + 1, size=(2, n))
    # (x, y) order if 1 in defs [0] and op [1], respectively
    ords = np.random.randint(2, size=(2, n))
    # index of ['+', '-', '*']
    ops = np.array(['+', '-', '*'])
    ops.reshape((1, 3))
    ops = ops[np.random.randint(3, size=n)]
    for i in range(n):
        x, y = vals[:, i]
        res = f'x={x},y={y}:' if ords[0, i] else f'y={y},x={x}:'
        o = ops[i]
        res += (f'x{o}y:' if ords[1, i] else f'y{o}x:')
        if o == '+':
            res += f'{x + y}'
        elif o == '*':
            res += f'{x * y}'
        else:
            assert o == '-'
            res += (f'{x - y}' if ords[1, i] else f'{y - x}')
        yield res


# tf.data.Dataset - abstraction for a sequence of elements
# each element is one or more Tensors

# can be consumed as iterables or as aggregatables (reduce)


def dset_gen_src(ps):
    # from_generator
    ds = td.Dataset.from_generator(
        lambda: native_source(ps),
        tf.string,
        tf.TensorShape([]),
    )
    return ds


def dset_src(ps):
    # range, from_tensor_slices, from_tensors
    ds = [s for s in native_source(ps)]
    ds = td.Dataset.from_tensor_slices(np.array(ds))
    return ds


def dset_filter(x):
    r = tf.strings.length(x) < 15
    tf.print(tf.strings.format('filtering {}... ', x) + ('in' if r else 'out'))
    return r


@tf.function
def dset_features(x):
    fs = tf.strings.split(x, ':')
    return {'defs': fs[0], 'op': fs[1], 'res': fs[2]}


tokens = {k: v for v, k in enumerate(vocab, start=5)}


@tf.function
def dset_tokenize(d):
    return {
        k: tf.numpy_function(
            lambda x: tf.constant([tokens[chr(c)] for c in x]),
            [d[k]],
            Tout=tf.int32,
        )
        for k in ('defs', 'op', 'res')
    }


# potentially large
# inside a tf.function or eagerly
# transparent performance is key


def main(_):
    ps = Params(**params)
    for s in native_source(ps):
        print(s)
    # cache, concatenate, enumerate, reduce, repeat, shuffle, skip, take, zip
    print('Ops on datasets')
    dg = dset_gen_src(ps)
    for s in dg.take(2):
        print(s)
    ds = dset_src(ps)
    for i, s in ds.take(2).concatenate(dg).enumerate():
        print(i, s)
    # filter
    print('Filter dataset elements')
    for i, s in enumerate(ds.filter(dset_filter)):
        print(s)
    # map
    for s in ds.map(dset_features).take(1):
        print(s)
    for s in ds.map(dset_features).map(dset_tokenize).take(1):
        print(s)
    # apply, flat_map, interleave

    # batch, padded_batch

    # prefetch


if __name__ == '__main__':
    from absl import app
    app.run(main)
