import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 40
trX = np.linspace(-1, 1, 101)
index = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(index):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5
X = tf.placeholder("float")
Y = tf.placeholder("float")
def model(X,w):
  temps = []
  for i in range(index):
    temp = tf.multiply(w[i],tf.pow(X,i))
    temps.append(temp)
  return tf.add_n(temps)
w = tf.Variable([0.] * index,name = "parameters")
y_pred = model(X,w)
cost = tf.reduce_sum(tf.square(Y-y_pred))
optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(training_epochs):
    for (x, y) in zip(trX, trY):
        sess.run(optimize, feed_dict={X: x, Y: y})

  w_val = sess.run(w)
  print(w_val)
  sess.close()
plt.scatter(trX,trY)
trY2 = 0
for i in range(index):
  trY2 += w_val[i] * np.power(trX, i)
plt.plot(trX, trY2, 'r')
plt.show()
