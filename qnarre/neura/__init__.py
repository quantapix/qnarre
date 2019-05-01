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

Conv1D = tf.keras.layers.Conv1D
Adam = tf.keras.optimizers.Adam
Dense = tf.keras.layers.Dense
Embedding = tf.keras.layers.Embedding
Event = tf.compat.v1.Event
GradientTape = tf.GradientTape
Input = tf.keras.Input
L1L2 = tf.keras.regularizers.L1L2
Layer = tf.keras.layers.Layer
Model = tf.keras.Model
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
Relu = tf.keras.activations.relu
SparseCategoricalAccuracy = tf.keras.metrics.SparseCategoricalAccuracy
SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy
Tanh = tf.keras.activations.tanh
TensorBoard = tf.keras.callbacks.TensorBoard
TensorShape = tf.TensorShape
TruncatedNormal = tf.keras.initializers.TruncatedNormal
abs = tf.keras.backend.abs
arange = tf.keras.backend.arange
argmax = tf.argmax
as_dtype = tf.as_dtype
cast = tf.keras.backend.cast
cast_to_floatx = tf.keras.backend.cast_to_floatx
concatenate = tf.keras.backend.concatenate
constant = tf.keras.backend.constant
cos = tf.keras.backend.cos
create_file_writer = ts.create_file_writer
cumsum = tf.keras.backend.cumsum
dot = tf.keras.backend.dot
equal = tf.equal
exp = tf.keras.backend.exp
expand_dims = tf.keras.backend.expand_dims
float16 = tf.float16
float32 = tf.float32
floatx = tf.keras.backend.floatx
function = tf.function
greater = tf.keras.backend.greater
import_event = None  # ts.import_event
int_shape = tf.keras.backend.int_shape
is_built_with_cuda = tf.test.is_built_with_cuda
l2_normalize = tf.keras.backend.l2_normalize
log1p = tf.math.log1p
matmul = tf.matmul
mean = tf.keras.backend.mean
moments = tf.nn.moments
one_hot = tf.keras.backend.one_hot
permute_dimensions = tf.keras.backend.permute_dimensions
pow = tf.keras.backend.pow
print = tf.print
reshape = tf.keras.backend.reshape
scalar = ts.scalar
sin = tf.keras.backend.sin
softmax = tf.keras.activations.softmax
sqrt = tf.keras.backend.sqrt
square = tf.keras.backend.square
squeeze = tf.keras.backend.squeeze
sum = tf.keras.backend.sum
tanh = tf.keras.backend.tanh
where = tf.where
