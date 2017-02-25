from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def model(inputs, batch_size, train):
    with tf.name_scope('model'):
        splits = tf.unstack(inputs, axis=1)
        # results = map(lambda e: _submodel(e, batch_size), splits)

        results = []
        for i, e in enumerate(splits):
            with tf.name_scope("submodel_%d" % i):
                r = _submodel(e, batch_size=batch_size)
                results.append(r)

        results = tf.stack(results, axis=1)
        flat = tf.reshape(results, [batch_size, 240*13])
        if(train):
            flat = tf.nn.dropout(flat, keep_prob=0.5)
        return slim.fully_connected(flat, 600) # classify 500


def _submodel(images, batch_size): #imgsize = 30x30
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.elu):
        l1 = slim.conv2d(images, 10, [3, 3], padding='VALID') #28x28
        p1 = slim.max_pool2d(l1, [2, 2]) #14x14 #model/submodel_n/MaxPool2D/MaxPool:0
        l2 = slim.conv2d(p1, 20, [3, 3], padding='VALID') #12x12
        p2 = slim.max_pool2d(l2, [2, 2]) #6x6
        l3 = slim.conv2d(p2, 30, [3, 3], padding='VALID') #4x4
        p3 = slim.max_pool2d(l3, [2, 2]) #2x2
        l3 = slim.conv2d(p3, 30, [2, 2], activation_fn=None) #2x2
        flat = tf.concat([tf.reshape(p3, [batch_size, -1]) , tf.reshape(l3, [batch_size, -1])], 1)
        print(flat.shape)
        flat2 = slim.fully_connected(flat, 240)

    return flat2
