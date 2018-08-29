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
  n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
  n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
  Z = np.zeros((m, n_H, n_W, n_C))
  A_prev_pad = zero_pad(A_prev,pad)
  for i in range(m):
    a_prev_pad = A_prev_pad[i]
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          vert_start = h * stride
          vert_end = vert_start + f
          horiz_start = w * stride
          horiz_end = horiz_start + f
          a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
          #[X0->Xend,Y0->Yend,Z0->Zend]
          Z[i,h,w,c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
  assert(Z.shape == (m, n_H, n_W, n_C))
  cache = (A_prev, W, b, hparameters)
  return Z, cache
def pool_forward(A_prev,hparameters,mode = "max"):
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  stride = hparameters['stride']
  f = hparameters['f']
  n_H = int((n_H_prev - f ) / stride) + 1
  n_W = int((n_W_prev - f ) / stride) + 1  
  n_C = n_C_prev
  A = np.zeros((m, n_H, n_W, n_C))
  for i in range(m):
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          vert_start = h * stride
          vert_end   = vert_start + f
          horiz_start= w * stride
          horiz_end  = horiz_start + f
          a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
          if(mode == "max"):
            A[i,h,w,c] =  np.max(a_prev_slice)
          elif (mode == "average"):
            A[i,h,w,c] = np.mean(a_prev_slice)
  cache = (A_prev,hparameters)
  assert(A.shape == (m,n_H,n_W,n_C))
  return A,cache
#29/8 
def conv_backward(dZ,cache):
  (A_prev, W, b, hparameters) = cache
  (f, f, n_C_prev, n_C) = W.shape
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  stride = hparameters["stride"]
  pad = hparameters["pad"]
  (m, n_H, n_W, n_C) = dZ.shape
  dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
  dW = np.zeros((f, f, n_C_prev, n_C))
  db = np.zeros((1, 1, 1, n_C))
  A_prev_pad = zero_pad(A_prev, pad)
  dA_prev_pad = zero_pad(dA_prev, pad)
  for i range(m):
    a_prev_pad = A_prev_pad[i]
    da_prev_pad = dA_prev_pad[i]
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          horiz_start = w* stride
          horiz_end   = horiz_start + f
          vert_start  = h * stride
          vert_end    = vert_start + f          
          a_slice     = a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
          da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
          dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
          db[:,:,:,c] += dZ[i, h, w, c]                
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db
