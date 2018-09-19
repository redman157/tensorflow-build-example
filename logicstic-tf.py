import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 1000

x_train = np.random.rand(1,100)
y_train = np.random.rand(1,100)
def linear_model(X,w,b):
  return tf.add(tf.matmul(X,w),b)
plt.scatter(x_train,y_train)
X = tf.placeholder("float",name = "x")
Y = tf.placeholder("float",name = "y")
w = tf.Variable(tf.random_normal([4,1],mean =0.0,name="parameters",stddev = 0.0))
b = tf.Variable([0.])

y_pred_sigmoid = tf.sigmoid(w[1] *X +w[0])
cost = tf.reduce_mean(-Y * tf.log(y_pred) - (1-Y) * tf.log(1-y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
delta = tf.abs((Y - y_pred_sigmoid))
correct_pred = tf.cast(tf.less(delta, tf.constant(0.5)), tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(training_epochs):
    optimizer.run(fd_train, feed_dict={batch_xs: Y,batch_ys:y_pred_sigmoid})                 
        
    if i % 500 == 0:
      loss_step = loss.eval(fd_train)
      train_accuracy = accuracy.eval(fd_train)
      print('  step, loss, accurary = %6d: %8.3f,%8.3f' % (i, 
                                                loss_step, train_accuracy))      
      fd_test = {x: te_x, y_: te_y.reshape((-1, 1))}
      print('accuracy = %10.4f' % accuracy.eval(fd_test))    
    

