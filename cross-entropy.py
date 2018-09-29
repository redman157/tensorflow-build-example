from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
FLAGS = None
def main(_,learning_rate,batch_size):
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_pred = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.matmul(y,tf.log(y_pred),tf.matmul(tf.subtract(1,y),tf.log(tf.subtract(1,y_pred))))))
  train_steps = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  for i in range(batch_size):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_pred = tf.equal(tf.argmax(y,1),y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print(sess.run(accuracy,feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels
    }))
