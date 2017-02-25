# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 30,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of ImageNet images for evaluation.

  Use this function as the inputs for evaluating a network.

  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping and resizing of the image to fit the network.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    labels: 1-D integer Tensor of [FLAGS.batch_size].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)

  return images, labels


def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of distorted versions of ImageNet images.

  Use this function as the inputs for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
  return images, labels

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  # with tf.op_scope([image], scope, 'distort_color'):
  with tf.name_scope(scope, 'distort_color', [image]):
    color_ordering = thread_id % 2

    #image = _whitebalanced_image(image)
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    image = tf.clip_by_value(image, 0.0, 1.0)
    # image = tf.image.per_image_standardization(image)
    # The random_* ops do not necessarily clamp.
    return image

def distort_image(image, height, width, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    distorted_image = image
    resize_method = thread_id % 4
    distorted_image = distort_color(distorted_image, thread_id)

    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  # with tf.op_scope([image, height, width], scope, 'eval_image'):
  with tf.name_scope(scope, 'eval_image', [image,  height, width]):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image

def _flip_points(points, height, width):
        points = [width, 0] - points
        points = tf.abs(points)
        return points

def _ramdom_flip_image_point(image, points, train, scope=None):
    with tf.name_scope(scope, 'random_flip', [image, points]):
        random = tf.random_uniform([], minval=0, maxval=1)
        image = tf.cond(random < 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        points = tf.cond(random < 0.5, lambda: _flip_points(points, 230, 230), lambda: points)

    return [image, points]

def _crop_image(image, point, train, scope=None):
    with tf.name_scope(scope, 'crop_image', [image, point]):
        c = 39 if train else 29
        # bbox = [point[0] - 29, point[0]+29, point[1]-29, point[1]+29]
        image = tf.pad(image, [[c, c+1], [c, c+1], [0,0]])
        print(image)
        # image = tf.Print(image, [tf.shape(image)])
        # image = tf.image.pad_to_bounding_box(image, 0, 0, 250, 250)
        image = tf.image.crop_to_bounding_box(image, point[1], point[0], c*2+1, c*2+1)
        if(train):
            begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image),  bounding_boxes=[[[0.0, 0.0, 1.0, 1.0]]], area_range=[0.75, 1.0], aspect_ratio_range=[1.0, 1.0])
            image = tf.slice(image, begin, size)

    image = tf.image.resize_images(image, [30, 30])
    image.set_shape([30, 30, 3])
    return image

def extract_feature_points(image, points, train):
    image_block=[]
    
    for point in tf.unstack(points, axis=0):
        imaged = _crop_image(image, point, train)
        image_block.append(imaged)
    image = tf.stack(image_block)
    return image

def image_preprocessing(image_buffer, bbox, train, thread_id=0):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: JPEG encoded string Tensor
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if bbox is None:
      raise ValueError('Please supply a bounding box.')

    # image = decode_jpeg(image_buffer)
    # image = tf.reshape(image_buffer, [59, 59, -1])
    height = FLAGS.image_size
    width = FLAGS.image_size

    if train:
      image = distort_image(image_buffer, height, width, bbox, thread_id)
    else:
      # image = eval_image(image_buffer, height, width)
      image = image_buffer

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      points: 2-D integer Tensor of feature points extracted by dlib
      preprocessing. [2, 13]
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/data': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=""),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/points': tf.FixedLenFeature([], dtype=tf.string,
                                           default_value="")
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    return features['image/data'], features['image/class/label'], features['image/points'], features['image/class/text']


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
      dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers

    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].

    Raises:
      ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = tf.matching_files("./train/*") #dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                          shuffle=True,
                                                          capacity=16)
        else:
          filename_queue = tf.train.string_input_producer(data_files,
                                                          shuffle=False,
                                                          capacity=1)
        if num_preprocess_threads is None:
          num_preprocess_threads = FLAGS.num_preprocess_threads

        if num_preprocess_threads % 4:
          raise ValueError('Please make num_preprocess_threads a multiple '
                           'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
          num_readers = FLAGS.num_readers

        if num_readers < 1:
          raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        if train:
          examples_queue = tf.RandomShuffleQueue(
              capacity=min_queue_examples + 3 * batch_size,
              min_after_dequeue=min_queue_examples,
              dtypes=[tf.string])
        else:
          examples_queue = tf.FIFOQueue(
              capacity=examples_per_shard + 3 * batch_size,
              dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
          enqueue_ops = []
          for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

          tf.train.queue_runner.add_queue_runner(
              tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
          example_serialized = examples_queue.dequeue()
        else:
          reader = tf.TFRecordReader()
          _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, points, _ = parse_example_proto(
                example_serialized)
            # read back points
            with tf.name_scope('points_readout'):
              points = tf.decode_raw(points, out_type=tf.int64)
              points = tf.reshape(points, [13, 2])
              points = tf.to_int32(points)

            # read back images
            with tf.name_scope('images_readout'):
              image_buffer = tf.decode_raw(image_buffer, out_type=tf.uint8)
              image_buffer = tf.reshape(image_buffer, [230, 230, 3])
              image_buffer = tf.image.convert_image_dtype(image_buffer, tf.float32)

            if not thread_id: tf.summary.image('original_image', tf.expand_dims(image_buffer, 0))
            image_buffer, points = _ramdom_flip_image_point(image_buffer, points, train)
            if not thread_id: tf.summary.image('flipped_image', tf.expand_dims(image_buffer, 0))
            bbox = [0., 0., 1., 1.]
            image_buffer = image_preprocessing(image_buffer, bbox, train, thread_id)
            image = extract_feature_points(image_buffer, points, train)
            if not thread_id: tf.summary.image('divided_final', image)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = FLAGS.image_size
        width = FLAGS.image_size
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, 13, height, width, depth])

        # Display the training images in the visualizer.
        tf.summary.image('images', images[0])

        return images, tf.reshape(label_index_batch, [batch_size])
