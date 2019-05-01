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

import tensorflow as tf

from qnarre.neura.layers.ffn import ffns, DenseDense
from qnarre.neura.layers.attent import attents, DotAttent
from qnarre.neura.layers.norm import LayerNorm
from qnarre.neura.layers.embed import TokEmbed, TypEmbed, PosEmbed, PosTiming

Layer = tf.keras.layers.Layer
Embedding = tf.keras.layers.Embedding

__all__ = (
    DenseDense,
    DotAttent,
    LayerNorm,
    PosEmbed,
    PosTiming,
    TokEmbed,
    TypEmbed,
    attents,
    ffns,
)
