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

import qnarre.neura as Q
import qnarre.neura.utils as U

from qnarre.neura.layers.ffn import ffns
from qnarre.neura.layers.attent import attents
from qnarre.neura.layers.norm import LayerNorm, PreProc, PostProc
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming


class Trafo_old(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.tok_embed = TokEmbed(PS)
        self.enc_stack = Q.Dense(2 * PS.hidden_size, activation='relu')
        self.dec_stack = Q.Dense(PS.hidden_size, activation='relu')
        self.logits = Q.Dense(PS.vocab_size, activation=None)

    def build(self, input_shape):
        ctx, _, tgt = input_shape
        return super().build(input_shape)

    def call(self, inputs, training=None, **kw):
        ctx, _, tgt = inputs
        y = self.tok_embed(ctx, **kw)
        y = self.enc_stack(y, **kw)
        y = self.dec_stack(y, **kw)
        if training:
            print('training...')
        return self.to_logits(y, **kw)

    def to_logits(self, x, unks=None, prior=None, **kw):
        xs = Q.int_shape(x)
        y = Q.reshape(x, (-1, xs[-1]))
        y = self.logits(y, **kw)
        ys = Q.int_shape(y)
        y = Q.reshape(y, (-1, ) + xs[1:-1] + ys[-1:])
        if unks:
            y = Q.where(unks, y, prior)
        return y


