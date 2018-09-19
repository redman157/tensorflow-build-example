import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_0 = np.random.normal(5,1,10)
X_1 = np.random.normal(2,1,10)

xs  = np.append(X_0,X_1)
labels = [0.] * len(X_0) + [1.] * len(X_1)

learning_rate = 0.001
epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable([0., 0.], name="parameters")
def model(X,w):
  return tf.add(tf.multiply(w[1],tf.pow(X,1)),
                  tf.multiply(w[0],tf.pow(X,0)))
y_pred = model(X,w)
cost = tf.reduce_sum(tf.square(Y-y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(Y,tf.to_float(tf.greater(y_pred,0.5)))
print(correct_pred)
accuracy = tf.reduce_mean(tf.to_float(correct_pred))
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(epochs):
    sess.run(optimizer, feed_dict={X:xs,Y:labels})
    current_cost = sess.run(cost,feed_dict={X: xs, Y: labels})
  w_val = sess.run(w)
  print('learned parameters', w_val)
  print('accuracy', sess.run(accuracy, feed_dict={X: xs, Y: labels}))
sess.close()
all_xs = np.linspace(0, 10, 100)
plt.plot(all_xs, all_xs*w_val[1] + w_val[0])
plt.scatter(xs, labels)
plt.show()
