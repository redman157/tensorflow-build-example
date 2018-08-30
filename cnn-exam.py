import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
datadir = '/data'
num_iters = 1000
minibatch_size = 100
data = input_data.read_data_sets(datadir, one_hot = True)
x = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784,10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
lr = 0.1
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_pred, labels = y_true))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for step in range(num_iters):
    batch_xs, batch_ys = data.train.next_batch(minibatch_size)
    sess.run(optimizer, feed_dict={x:batch_xs, y_true:batch_ys})
        
  testing = sess.run(accuracy, feed_dict={x:data.test.images, y_true:data.test.labels})
print('ACcuracy: {:.4}'.format(testing*100))
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1,shape = shape)
  return tf.Variable(initial)
def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides= [1,1,1,1],padding = 'SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W)+b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W)+b
x = tf.placeholder(tf.float32,shape= [None,784])
y = tf.placeholder(tf.float32,shape= [None,10])
x_image = tf.reshape(x,[-1,28,28,1])
conv1 = conv_layer(x_image, shape=[5,5,1,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1,7*7*64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 10)
print(y_conv)
