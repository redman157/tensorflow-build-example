import numpy as np

learning_rate = 0.1
epoch =  10000
hidden_layer = 4
def sigmoid(x,deriv = False):
  if(deriv == True):
    return x*(1-x)
  return 1/(1 + np.exp(-x))

X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])
y = np.array([[0,1,1,0]])
np.random.seed(1)
weight_0 = 2*np.random.random((3,hidden_layer))-1
weight_1 = 2*np.random.random((hidden_layer,1))-1
for i in range(epoch):
  layer_0 = X
  layer_1 = sigmoid(np.dot(layer_0,weight_0))
  layer_2 = sigmoid(np.dot(layer_1,weight_1))
    
  layer_error_2 = (y - layer_2)
  delta_2 = layer_error_2 * sigmoid(layer_2,deriv = True)
  
  layer_error_1 = np.dot(delta_2,weight_1.T)
  delta_1 =  layer_error_1 * sigmoid(layer_1,deriv = True)
    
  weight_1 -= (learning_rate * layer_1.T.dot(delta_2))
  weight_0 -= (learning_rate * layer_0.T.dot(delta_1))
print("ket qua sau training",delta_1)
    

  
