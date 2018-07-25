import tensorflow as tf
import numpy as np

np.random.seed(1)
X = np.random.rand()
y = []
jHistory = tf.empty(epoch)
m = x.shape()
learning_rate = 0.1
y_pred = tf.matmul(X,theta)
def sigmoid(x, deriv = False):
  if True:
    return 1/(1+np.exp(-x))
  else:
    return x*(1-x)
def cost(x,learning_rate,theta,):
  return tf.reduce_sum(tf.pow(y_pred - y),2)/(2*m)
def lost_logistic(x,learning_rate,theta,sigmoid):

    return tf.reduce_sum(tf.pow(y_pred - y),2)/(2*m)
def NN(lost_logistic,sigmoid):
  
def optimizer(learning_rate,cost):
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
def gradent_descent(x,cost,epoch):
  for i in range(epoch):
    if epoch % 5 == 0:
      delta = tf.matmul(tf.pow(y_pred - y),2)
      theta -= learning_rate* delta
      jHistory[i] = cost(X,y,theta)
    return theta, jHistory
init = tf.global_variables_initializer()
