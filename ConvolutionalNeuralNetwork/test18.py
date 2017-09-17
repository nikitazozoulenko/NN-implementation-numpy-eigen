import numpy as np
from math_for_cnn import *
R= 1
D = 1
H = 3
W = 3
num_filters = 1
k_size = 2
stride = 1
pad = 0
new_width = int((W+pad*2-k_size)/stride + 1)




x = np.arange(R*D*H*W).reshape(R,D,H,W)
print("x",x)
WEIGHT = np.arange(4).reshape(1,1,2,2)
print("WEIGHT",WEIGHT)
im2rowx = im2row(x, size = k_size, stride = stride, pad = pad)
y = np.dot(im2rowx,WEIGHT.T.reshape((1*2*2, 1)))
print(y)
print(im2rowx)

w_reshaped = WEIGHT.reshape(D*k_size**2, num_filters)

test = np.dot(im2rowx, w_reshaped)
print("test",test)





self.im2rowx[i] = im2row(self.X[i], size = self.layers[i].kernel_size, stride = self.layers[i].stride, pad = self.layers[i].pad)
y = np.dot(self.im2rowx[i], self.W[i].T.reshape(( kernel_D*kernel_H*kernel_W , num_filters )))
self.X[i+1] = y.reshape((out_R, out_H, out_W, out_D)).transpose(0, 3, 1, 2)
