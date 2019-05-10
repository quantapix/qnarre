# Copyright 2018 Quantapix Authors. All Rights Reserved.
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

import json

import numpy as np

from imdb.params import PARAMS as PS


def prepare_data(base,
                 skip_top=0,
                 seed=113,
                 start_char=1,
                 oov_char=2,
                 index_from=3):
    (x_train, y_train), (x_test, y_test) = _load_numpy(base)

    indices = np.arange(len(x_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test, y_test = x_test[indices], y_test[indices]

    xs = np.concatenate([x_train, x_test])
    ys = np.concatenate([y_train, y_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    def _filterer(maxlen):
        for x, y in zip(xs, ys):
            if len(x) < maxlen:
                yield x, y

    if PS.maxlen:
        xs, ys = zip(*_filterer(PS.maxlen))
        if not xs:
            raise ValueError('Increase maxlen')

    count = PS.max_features or max([max(x) for x in xs])
    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < count) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < count] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(ys[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(ys[idx:])

    x_train = _pad_sequences(x_train)
    x_test = _pad_sequences(x_test)

    return (x_train, y_train), (x_test, y_test)


def load_word_index(base):
    p = base / 'imdb/np' / 'imdb_word_index.json'
    with open(p) as f:
        return json.load(f)


def _load_numpy(base):
    p = base / 'imdb/np' / 'imdb.npz'
    with np.load(p) as d:
        train = d['x_train'], d['y_train']
        test = d['x_test'], d['y_test']
    return train, test


def _pad_sequences(sequences,
                   dtype='int32',
                   pre_pad=True,
                   pre_truncate=True,
                   value=0.):
    count = len(sequences)
    maxlen = PS.maxlen or np.max([len(s) for s in sequences])
    shape = tuple()
    for s in sequences:
        if len(s) > 0:
            shape = np.asarray(s).shape[1:]
            break
    ss = (np.ones((count, maxlen) + shape) * value).astype(dtype)
    for i, s in enumerate(sequences):
        if len(s):
            trunc = s[-maxlen:] if pre_truncate else s[:maxlen]
            trunc = np.asarray(trunc, dtype=dtype)
            assert trunc.shape[1:] == shape
            if pre_pad:
                ss[i, -len(trunc):] = trunc
            else:
                ss[i, :len(trunc)] = trunc
    return ss
