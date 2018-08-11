import numpy as np
def gradient(x,Y,theta,cost,epoch):
  m = x.shape(0)
  for i in range(epoch):
    if(i % epoch == 0):
      print("gia tri cua %i theo %epoch ",i, epoch)
      theta = (-np.sum(cost * theta))/m
  return theta
def function(val,gradient)
def predict(w,b,X):
  m = X.shape[1] # thiet lap so luong chieu m(,X.shape)
  y_pred = np.zeros((1,m)) #y_pred = [1,m]
  w = w.shape(X.shape[0],1) # thiet lap so luong chieu ma tran w = [X.shape]
  A = sigmoid(np.dot(w.t,X)+b)
  for i in range(A.shape[1]):
    if(A[0][i] <= 0.5):
      Y_prediction[0][i] = 0
    else:
      Y_prediction[0][i] = 1
  assert(Y_prediction.shape == (1, m))
    
  return Y_prediction
print("predictions = " + str(predict(w, b, X)))
