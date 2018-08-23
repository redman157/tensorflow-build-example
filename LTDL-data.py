import tensorflow as tf
import pandas as pd 
import argparse
import numpy as np

np.random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
index = 6 
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
conv_layers = {}
# dinh dang kieu du lieu cho X va y lan luot theo height,weight va color
def create_placeholders(n_H0, n_W0, n_C0, n_y):
  X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
  Y = tf.placeholder(tf.float32, [None, n_y])
  return X,Y
def initialize_parameters():
  tf.set_random_seed(1)
  # tao ra trong so voi so chieu cua W [x,y,z, so ma tran]
  W1 = tf.get.variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed = 0)
  W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
   parameters = {"W1": W1,
                  "W2": W2}
  return parameters
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
  
					   
