import  numpy  as np
import  h5py
import  matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
def zero_pad(X,pad):
  X_pad = np.pad(X,((0, 0), (pad, pad), (pad, pad), (0, 0)),'constant',constant_values=0)
  return X_pad
def conv_single_step(A_prev,W,b):
  s = np.multiply(A_prev,W)+b
  # s = (A*W)+b
  Z = np.sum(s)
  return Z
def conv_forward(A_prev, W, b, hparameters):
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  (f,f,n_C_prev,n_C) = W.shape
  stride = hparameters['stride']
  pad = hparameters['pad']

  
