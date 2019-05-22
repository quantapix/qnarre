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

import pickle

import numpy as np
import tensorflow as tf

from keras import backend, utils

from cifar.params import PARAMS as PS


def prepare_data(base):
    if PS.num_classes == 10:
        (x_train, y_train), (x_test, y_test) = _cifar10(base)
    else:
        (x_train, y_train), (x_test, y_test) = _cifar100(base)

    x_train, x_test = map(lambda x: x.astype('float32') / 255,
                          (x_train, x_test))
    y_train, y_test = map(lambda y: utils.to_categorical(y, PS.num_classes),
                          (y_train, y_test))

    if backend.image_data_format() == 'channels_last':
        x_train, x_test = map(lambda x: x.transpose(0, 2, 3, 1),
                              (x_train, x_test))

    return (x_train, y_train), (x_test, y_test)


def prepare_dataset(base):
    if PS.num_classes == 10:
        return tuple(_cifar10_dataset(base))
    return tuple(_cifar100_dataset(base))


_SHAPE = (3, 32, 32)
_FLIPPED = (32, 32, 3)


def _cifar10(base):
    batches, per = 5, 10000
    samples = batches * per

    x_train = np.empty((samples, *_SHAPE), dtype='uint8')
    y_train = np.empty((samples, 1), dtype='uint8')

    b = base / 'cifar' / 'cifar-10-batches-py'
    for i in range(batches):
        s, e = i * per, (i + 1) * per
        p = b / 'data_batch_{}'.format(i + 1)
        x_train[s:e, :, :, :], y_train[s:e, :] = _load_batch(p)

    test = _load_batch(p / 'test_batch')

    return (x_train, y_train), test


def _cifar100(base, kind):
    assert kind in ('fine', 'coarse')
    p = base / 'cifar' / 'cifar-100-python'

    train = _load_batch(p / 'train', kind + '_labels')
    test = _load_batch(p / 'test', kind + '_labels')

    return train, test


def _load_batch(path, labels='labels'):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    d = {k.decode(): v for k, v in d.items()}
    return d['data'].reshape(-1, *_SHAPE), d[labels].reshape(-1, 1)


def _cifar10_dataset(base, distort):
    p = base / 'cifar/tf'
    for n in ('train', 'test'):
        n = [str(p / ('cifar-10-' + n + '.tfrecords'))]
        ds = tf.data.TFRecordDataset(n).repeat()

        def _single(serialized):
            fs = tf.parse_single_example(
                serialized,
                features={
                    'x': tf.FixedLenFeature([], tf.string),
                    'y': tf.FixedLenFeature([], tf.int64),
                })
            x = tf.decode_raw(fs['x'], tf.uint8)
            x = tf.cast(x, tf.float32)
            # x.set_shape(_SHAPE)
            x = tf.reshape(x, _SHAPE)
            x = tf.transpose(x, [1, 2, 0])
            if distort and n == 'train':
                x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
                x = tf.random_crop(x, _FLIPPED)
                x = tf.image.random_flip_left_right(x)

            y = tf.cast(fs['y'], tf.int32)
            return x, y

        ds = ds.map(_single, num_parallel_calls=PS.batch_size)
        if n == 'train':
            ds = ds.shuffle(PS.buffer_size)
        yield ds.batch(PS.batch_size)


def _cifar100_dataset(base):
    pass


IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(dir_data, batch_size):

    filenames = [
        os.path.join(dir_data, 'data_batch_%d.bin' % i) for i in range(1, 6)
    ]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(
            distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(
        float_image,
        read_input.label,
        min_queue_examples,
        batch_size,
        shuffle=True)
