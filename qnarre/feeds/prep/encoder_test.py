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

import zipfile

import qnarre.neura.utils as U

from qnarre.feeds.prep import utils
from qnarre.feeds.prep import encoder

ps = dict(
    lower_case=True,
    model='',
    tok_max_chars=None,
    vocab_pairs=None,
    bert_vocab='.model/bert/uncased_L-12_H-768_A-12/vocab.txt',
    gpt_2_vocab='.model/gpt_2/117M/encoder.json',
    gpt_2_pairs='.model/gpt_2/117M/vocab.bpe',
)

ps = U.Params(ps)


def test_encoders():
    txt = "sf!fg dfg'sdf?dfg xcxb'sdfg!sdg 324sdf.sdfa"
    ce = encoder.CharE(ps)
    ts, os, _ = zip(*ce(txt))
    d = ce.decode(ts, os)
    assert d == txt
    we = encoder.WordE(ps)
    ts, os, _ = zip(*we(txt))
    d = we.decode(ts, os)
    assert d == txt
    be = encoder.BertE(ps)
    ge = encoder.Gpt2E(ps)
    with zipfile.ZipFile('.data/text8/text8.zip') as z:
        with z.open('text8') as f:
            ws = utils.normalize(f.read().decode().strip())
            ws = utils.normalize(ws).split()
            for i in range(200):
                txt = ' '.join(ws[i * 100:i * 100 + 100])
                ts, os, _ = zip(*ce(txt))
                d = ce.decode(ts, os)
                assert d == txt
                ts, os, _ = zip(*we(txt))
                d = we.decode(ts, os)
                assert d == txt
                ts, os, _ = zip(*be(txt))
                d = be.decode(ts, os)
                assert d == txt
                ts, os, _ = zip(*ge(txt))
                d = ge.decode(ts, os)
                assert d == txt
    print(len(ce.vocab), len(we.vocab), len(be.vocab), len(ge.vocab))
