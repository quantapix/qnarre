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
import tensorflow.summary as ts
import tensorflow_probability as tfp

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op

ks = tf.keras
K = ks.backend

Adam = ks.optimizers.Adam
BatchNormalization = ks.layers.BatchNormalization
Conv1D = ks.layers.Conv1D
Dense = ks.layers.Dense
Dropout = ks.layers.Dropout
Embedding = ks.layers.Embedding
Event = tf.compat.v1.Event
GradientTape = tf.GradientTape
Input = ks.Input
L1L2 = ks.regularizers.L1L2
Layer = ks.layers.Layer
Model = ks.Model
ModelCheckpoint = ks.callbacks.ModelCheckpoint
Relu = ks.activations.relu
Softmax = ks.activations.softmax
SparseCategoricalAccuracy = ks.metrics.SparseCategoricalAccuracy
SparseCategoricalCrossentropy = ks.losses.SparseCategoricalCrossentropy
Tanh = ks.activations.tanh
TensorBoard = ks.callbacks.TensorBoard
TensorShape = tf.TensorShape
TruncatedNormal = ks.initializers.TruncatedNormal
abs = math_ops.abs
argmax = math_ops.argmax
argmin = math_ops.argmin
as_dtype = tf.as_dtype
bias_add = K.bias_add
bitcast = array_ops.bitcast
bitwise_and = tf.bitwise.bitwise_and
bitwise_or = tf.bitwise.bitwise_or
bitwise_xor = tf.bitwise.bitwise_xor
bool = tf.bool
cast = math_ops.cast
cast_to_floatx = K.cast_to_floatx
concat = array_ops.concat
constant = constant_op.constant
cos = math_ops.cos
create_file_writer = ts.create_file_writer
cumprod = math_ops.cumprod
cumsum = math_ops.cumsum
dot = K.dot
equal = math_ops.equal
exp = math_ops.exp
expand_dims = array_ops.expand_dims
fill = array_ops.fill
float16 = tf.float16
float32 = tf.float32
floatx = K.floatx
function = tf.function
gather = array_ops.gather
gather_nd = array_ops.gather_nd
greater = math_ops.greater
greater_equal = math_ops.greater_equal
identity = array_ops.identity
import_event = None  # ts.import_event
int32 = tf.int32
int_shape = K.int_shape
is_built_with_cuda = tf.test.is_built_with_cuda
is_nan = math_ops.is_nan
l2_normalize = tf.nn.l2_normalize
less = math_ops.less
less_equal = math_ops.less_equal
log1p = math_ops.log1p
log_softmax = tf.nn.log_softmax
logical_and = math_ops.logical_and
logical_not = math_ops.logical_not
logical_or = math_ops.logical_or
logical_xor = math_ops.logical_xor
matmul = math_ops.matmul
moments = tf.nn.moments
multinomial = tfp.distributions.multinomial
one_hot = array_ops.one_hot
ones = array_ops.ones
ones_like = array_ops.ones_like
pow = math_ops.pow
print = tf.print
range = math_ops.range
reduce_all = math_ops.reduce_all
reduce_any = math_ops.reduce_any
reduce_logsumexp = math_ops.reduce_logsumexp
reduce_max = math_ops.reduce_max
reduce_mean = math_ops.reduce_mean
reduce_min = math_ops.reduce_min
reduce_prod = math_ops.reduce_prod
reduce_sum = math_ops.reduce_sum
reshape = array_ops.reshape
scalar = ts.scalar
sin = math_ops.sin
softmax = tf.nn.softmax
sqrt = math_ops.sqrt
square = math_ops.square
squeeze = array_ops.squeeze
stack = array_ops.stack
tanh = math_ops.tanh
tensor_scatter_nd_update = tf.tensor_scatter_nd_update
tile = array_ops.tile
top_k = tf.nn.top_k
transpose = array_ops.transpose
unstack = array_ops.unstack
where = array_ops.where
zeros = array_ops.zeros
