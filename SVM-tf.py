from __future__  import division
import argparse
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()

X_val = np.array(iris.data[ : , : ])
y_val = np.array(iris.target[:])
epsilon = tf.constant([0.5])

train_indices = np.random.choice(len(X_val), round(len(X_val)*0.8), replace=False)
test_indices = np.array(list(set(range(len(X_val))) - set(train_indices)))



X_train = X_val[train_indices]
y_train = y_val[train_indices]
X_test = X_val[test_indices]
y_test = y_val[test_indices]

batch_size = 50
learning_rate = 0.075

X_data = tf.placeholder(shape=[None,1],dtype= tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype = tf.float32)

W = tf.Variable(tf.random_normal(shape= [1,1]))
b = tf.Variable(tf.random_normal(shape= [1,1]))

def model(X,W,b):
  return tf.add(tf.matmul(X,W),b)
y_pred = model(X_data,W,b)
loss = tf.reduce_mean(tf.maximum(0.,tf.subtract(tf.abs(tf.subtract(y_pred , y_target)),epsilon)))

optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_pred = tf.equal(y_target,y_pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  temp_train = []
  temp_test  = [] 
  for i in range(200):
    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = np.transpose([X_train[rand_index]])
    rand_y = np.transpose([y_train[rand_index]])
    
    sess.run(optimize,feed_dict={X_data: rand_x,y_target: rand_y})
    
    train = sess.run(loss, feed_dict={X_data : np.transpose(X_train), y_target : np.transpose(y_train) })
    temp_train.append(train)
    
    test  = sess.run(loss, feed_dict={X_data : np.transpose(X_test), y_target : np.transpose(y_test) })
    temp_test.append(test)
    acc_train = sess.run(accuracy,feed_dict={y_pred : y_pred, y_target : y_train})
    acc_test = sess.run(accuracy,feed_dict={y_pred : y_pred, y_target : y_test})
[[slope]] = sess.run(W)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)
