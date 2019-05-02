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
abs = K.abs
arange = K.arange
argmax = tf.argmax
as_dtype = tf.as_dtype
bias_add = K.bias_add
cast = K.cast
cast_to_floatx = K.cast_to_floatx
concatenate = K.concatenate
constant = K.constant
cos = K.cos
create_file_writer = ts.create_file_writer
cumsum = K.cumsum
dot = K.dot
equal = tf.equal
exp = K.exp
expand_dims = K.expand_dims
float16 = tf.float16
float32 = tf.float32
floatx = K.floatx
function = tf.function
gather = tf.gather
greater = K.greater
import_event = None  # ts.import_event
int_shape = K.int_shape
is_built_with_cuda = tf.test.is_built_with_cuda
l2_normalize = K.l2_normalize
log1p = tf.math.log1p
log_softmax = tf.nn.log_softmax
matmul = tf.matmul
mean = K.mean
moments = tf.nn.moments
one_hot = K.one_hot
permute_dimensions = K.permute_dimensions
pow = K.pow
print = tf.print
reshape = K.reshape
scalar = ts.scalar
sin = K.sin
sqrt = K.sqrt
square = K.square
squeeze = K.squeeze
sum = K.sum
tanh = K.tanh
transpose = tf.transpose
unstack = tf.unstack
where = tf.where
logical_not = tf.math.logical_not
tensor_scatter_nd_update = tf.tensor_scatter_nd_update
