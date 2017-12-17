#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn16_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

fname = sys.argv[1]

img1 = scp.misc.imread(fname)

sess = tf.InteractiveSession()
images = tf.placeholder("float")
feed_dict = {images: img1}
batch_images = tf.expand_dims(images, 0)

vgg_fcn = fcn16_vgg.FCN16VGG()
with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images, debug=True)

print('Finished building Network.')

init = tf.global_variables_initializer()
sess.run(init)

print('Running the Network')
down = sess.run(vgg_fcn.pred, feed_dict=feed_dict)

segmentation_downsampled = utils.color_image(down[0])
