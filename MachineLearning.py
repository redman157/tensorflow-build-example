import tensorflow as tf
import numpy as np

np.random.seed(1)
X = np.random.rand()
y = []
jHistory = tf.empty()
m = x.shape()
learning_rate = 0.1
def sigmoid(x, deriv = False):
  if True:
    return 1/(1+np.exp(-x))
  else:
    return x*(1-x)
def cost(x,learning_rate,bias,):
  return tf.reduce_sum(tf.pow(y_pred - y),2)/(2*m)
def optimizer(learning_rate,cost):
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
def gradent_descent(x,cost,epoch):
  for i in range(epoch):
    if epoch % 5 == 0:
      c = 
