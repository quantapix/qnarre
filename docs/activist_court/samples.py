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

import utils as qu

randint = np.random.randint

np.random.seed(12345)


class Samples:
    def __init__(self, ps):
        self.ps = ps
        mx, n = ps.max_val, ps.dim_pool
        self.xys = randint(low=1 - mx, high=mx, size=(2, n))
        self.seq = randint(2, size=(2, n))
        self.ops = np.array(['+', '-', '*'])[randint(3, size=n)]
        self.yns = randint(2, size=(2, n))
        self.idx = 0

    @property
    def next_idx(self):
        self.idx += 1
        if self.idx >= self.ps.dim_pool:
            self = Samples(self.ps)
        return self, self.idx

    def create(self, i, use_x=None):
        if use_x is not None:
            self.xys[0, i] = use_x
        x, y = self.xys[:, i]
        if use_x is None:
            enc = f'x={x},y={y}' if self.seq[0, i] else f'y={y},x={x}'
        else:
            enc = f'x=$,y={y}' if self.seq[0, i] else f'y={y},x=$'
        o = self.ops[i]
        enc += ';' + (f'x{o}y' if self.seq[1, i] else f'y{o}x')
        if o == '+':
            res = x + y
        elif o == '*':
            res = x * y
        else:
            assert o == '-'
            res = (x - y) if self.seq[1, i] else (y - x)
        return enc, res

    def other_than(self, x):
        mx = self.ps.max_val
        while True:
            y = randint(low=1 - mx, high=mx)
            if y != x:
                return y


def mask(x):
    x, lx = list(x), len(x)
    for i in randint(lx, size=(lx // 2)):
        x[i] = '?'
    return ''.join(x)


def sampler(ps):
    ss = Samples(ps)
    for _ in range(ps.num_samples):
        ss2 = None
        ss, idx = ss.next_idx
        enc, res = ss.create(idx)
        dec = tgt = f'[{res}]'
        yn = ss.yns[0, idx]
        yns = dict(tgt=yn)
        if not yn:
            bad = f'[{ss.other_than(res)}]'
            yns.update(dec=bad)
        ss2, i2 = ss.next_idx
        e2, r2 = ss2.create(i2, x=res)
        d2 = e2 + f'[{r2}]'
        ynx = dict(enc=enc + tgt, dec=d2, tgt=yn)
        if not yn:
            if randint(2):
                ynx.update(dec=e2 + f'[{ss2.other_than(r2)}]')
            else:
                ynx.update(enc=enc + bad)
        msk = dict(dec=mask(dec))
        msx = dict(enc=enc + tgt, dec=mask(d2))
        pre, post = f'{ss2.other_than(res)}', f'{ss2.other_than(res)}'
        dqa = f'[{pre}{res}{post}]'
        qas = dict(dec=dqa, tgt=[len(pre) + 1, len(dqa) - len(post) - 1])
        yield {
            'enc': enc,
            'dec': dec,
            'tgt': tgt,
            'yns': yns,
            'ynx': ynx,
            'msk': msk,
            'msx': msx,
            'qas': qas,
        }
        ss = ss2


params = dict(
    dim_pool=8 * 1024,
    max_val=1000,
    num_samples=1000,
)


def main(ps):
    pass


if __name__ == '__main__':
    ps = qu.Params(**params)
    main(ps)
