import numpy as np

learning_rate = 0,1
epoch =  5000
def sigmoid(x,deriv = False):
  if(deriv == True):
    return x*(1-x)
  return 1/(1 + np.exp(-x))
np.random.seed(1)
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])
y = np.array([[0,0,1,1]]).T
weight_0 = 2*np.random.random((3,hi))-1
weight_1 = 2*np.random.random((hidden_layer,1))-1
for i in range(epoch):
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0,weight))
    layer_error = (y - layer_1)
    delta = layer_error * sigmoid(layer_1,True)
    weight += np.dot(layer_0.T,delta)
print("ket qua sau training",layer_1)
    

  
