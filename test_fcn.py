#!/usr/bin/env python

import scipy.misc
import tensorflow as tf
import sys
import fcn32_vgg
import utils

fname = sys.argv[1]

img1 = scipy.misc.imread(fname)

sess = tf.InteractiveSession()
images = tf.placeholder("float")
feed_dict = {images: img1}
batch_images = tf.expand_dims(images, 0)

vgg_fcn = fcn32_vgg.FCN32VGG()
with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images)

print('Finished building Network.')

init = tf.global_variables_initializer()
sess.run(init)

print('Running the Network')
output = sess.run(vgg_fcn.pred_up, feed_dict=feed_dict)
