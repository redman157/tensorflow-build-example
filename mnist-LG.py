import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
def model(X,w,b):
  return (tf.matmul(X,w) + b)

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
X_train,Y_train,X_test,Y_test = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

learning_rate = 0.01
training_epochs = 20
batch_size = 1000
display_step = 1

X = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10 ])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

Y_pred = tf.nn.softmax(model(X,w,b)) # y = tf.log(X*w + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(Y_pred), reduction_indices=1)) # y_pred / y_pred*sumy
optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)
pred_op = tf.argmax(Y_pred,1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  avg_cost = 0.
  for i in range(batch_size):
    for start,end in zip(range(0,len(X_train),128),range(128, len(X_train)+1, 128)):
      
      c =sess.run([optimizer], feed_dict={X:X_train, y : Y_train})
      avg_cost += c/batch_size
  w_val = sess.run(w)
  
  if (epoch+1) % display_step == 0:
     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
  correct_pred = tf.equal(tf.argmax(Y_pred,1),tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Accuracy:", accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

