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

randint = np.random.randint

groups = ('yns', 'ynx', 'msk', 'msx', 'cls', 'clx', 'qas', 'rev', 'gen', 'fix')


def sampler(ps):
    ss = Samples(ps)
    for _ in range(ps.num_samples):
        ss2 = None
        ss, idx = ss.next_idx
        enc, res, *_ = ss.create(idx)
        dec = tgt = f'[{res}]'
        bad = f'[{ss.other(res)}]'
        yn = ss.yns[0, idx]

        yns = dict(enc=enc, dec=dec if yn else bad, tgt=f'#{yn}')

        ss2, i2 = ss.next_idx
        e2, r2, *_ = ss2.create(i2, use_x=res)
        d2 = e2 + f'[{r2}]'
        ynx = dict(enc=enc + tgt, dec=d2, tgt=f'#{yn}')
        if not yn:
            if randint(2):
                ynx.update(dec=e2 + f'[{ss2.other(r2)}]')
            else:
                ynx.update(enc=enc + bad)

        msk = dict(enc=enc, dec=mask(dec), tgt=tgt)

        msx = dict(enc=enc + tgt, dec=mask(d2), tgt=d2)

        t2 = np.array(['+', '*', '2'])[randint(3)]
        ss2, i2 = ss2.next_idx
        e2, r2, t2, _ = ss2.create(i2, double_y=(t2 == '2'))
        cls = dict(enc=e2, dec=f'[{r2}]', tgt=t2)

        t3 = np.array(['0', '+', '-'])[randint(3)]
        ss2, i2 = ss2.next_idx
        e2, r2, _, t3 = ss2.create(i2, use_x=None if t3 == '0' else res)
        clx = dict(enc=enc + tgt, dec=e2 + f'[{r2}]', tgt=t3)

        r1, r3 = f'{ss2.other(res)}', f'{ss2.other(res)}'
        r2 = f'[{r1}{res}{r3}]'
        b, e = len(r1) + 1, len(r2) - len(r3) - 1
        qas = dict(enc=enc, dec=r2, tgt=f'#{b} {e}')

        e2, r2, *_ = ss.create(idx, keep=False)
        d2 = e2 + f'[{r2}]' if yn else bad
        rev = dict(enc=enc + tgt, dec=d2, tgt=f'#{yn}')

        gen = dict(enc=enc, dec='[?', tgt=tgt)
        fix = dict(enc=enc, dec=mask(dec, '_'), tgt=tgt)

        yield {
            'yns': yns,
            'ynx': ynx,
            'msk': msk,
            'msx': msx,
            'cls': cls,
            'clx': clx,
            'qas': qas,
            'rev': rev,
            'gen': gen,
            'fix': fix,
        }
        ss = ss2


class Samples:
    def __init__(self, ps):
        self.ps = ps
        mx, n = ps.max_val, ps.dim_pool
        self.xys = randint(low=1 - mx, high=mx, size=(2, n))
        self.ops = np.array(['+', '-', '*'])[randint(3, size=n)]
        self.seq = randint(2, size=(2, n))
        self.yns = randint(2, size=(2, n))
        self.idx = 0

    @property
    def next_idx(self):
        self.idx += 1
        if self.idx >= self.ps.dim_pool:
            self = Samples(self.ps)
        return self, self.idx

    def create(self, i, use_x=None, double_y=None, keep=True):
        if use_x is not None:
            self.xys[0, i] = use_x
        op = self.ops[i]
        if double_y:
            self.xys[1, i] = 2 if op == '*' else self.xys[0, i]
        x, y = self.xys[:, i]
        keep ^= not (self.seq[0, i])
        if use_x is None:
            xys = f'x={x},y={y}' if keep else f'y={y},x={x}'
        else:
            xys = f'x=$,y={y}' if keep else f'y={y},x=$'
        xys += ';' + (f'x{op}y' if self.seq[1, i] else f'y{op}x')
        if op == '+':
            res = x + y
        elif op == '*':
            res = x * y
        else:
            assert op == '-'
            res = (x - y) if self.seq[1, i] else (y - x)
        cls = '2' if double_y else ('*' if op == '*' else '+')
        clx = '0' if use_x is None else ('+' if res >= use_x else '-')
        return xys, res, cls, clx

    def other(self, x):
        mx = self.ps.max_val
        while True:
            y = randint(low=1 - mx, high=mx)
            if y != x:
                return y


def mask(x, msk=None):
    x, lx = list(x), len(x)
    for i in randint(lx, size=((lx // 2) if msk is None else 1)):
        x[i] = msk or '?'
    return ''.join(x)


if __name__ == '__main__':
    np.random.seed(12345)
    import utils as qu
    ps = dict(
        dim_pool=5,
        max_val=100,
        num_samples=15,
    )
    ps = qu.Params(**ps)
    for d in sampler(ps):
        for g in groups:
            print(f'sample {g}:', d[g])
