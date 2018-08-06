import numpy as np
epoch = 10000
def sigmoid(x,deriv = False):
  if(deriv == True):
    return x*(1-x)
  return 1/(1+np.exp(-x))
X = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])
y = np.array([[0,0,0,1]]).T
np.random.seed(1)
weight = 2*np.random.random((2,1)) - 1
for i in range(epoch):
  layer_0 = X
  layer_1 = sigmoid(np.dot(layer_0,weight))
  error = layer_1 - y
  delta = error * sigmoid(layer_1,deriv = True)
  weight += np.dot(layer_0.T,delta)
print("ket qua sau training: ",layer_1)
