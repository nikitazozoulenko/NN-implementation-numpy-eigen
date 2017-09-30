import numpy as np
from math_for_cnn import *

R = 2
D = 3
H = 4
W = 4
x = np.arange(R*D*H*W).reshape(R,D,H,W)


x = np.array([[[[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]],

  [[16, 17, 18, 19],
   [20, 21, 22, 23],
   [24, 100, 26, 27],
   [28, 29, 30, 31]],

  [[32, 33, 34, 35],
   [36, 37, 38, 39],
   [40, 41, 42, 43],
   [44, 45, 46, 47]]],


 [[[48, 49, 50, 51],
   [200, 53, 54, 55],
   [56, 57, 58, 59],
   [60, 61, 62, 63]],

  [[64, 65, 66, 67],
   [68, 69, 70, 71],
   [72, 73, 74, 75],
   [76, 77, 78, 79]],

  [[80, 81, 82, 83],
   [84, 85, 86, 87],
   [88, 89, 300, 91],
   [92, 93, 94, 95]]]])
print(x)
x_reshaped = x.reshape(R * D, 1, H, W)

print(im2row(x, size = 2, stride = 2, pad = 0))
im2row_pool = im2row(x_reshaped, size = 2, stride = 2, pad = 0)
print(im2row_pool)

x_max_arg = np.argmax(im2row_pool, axis = 1)
print(x_max_arg)

out = im2row_pool[range(x_max_arg.size), x_max_arg]
print(out.shape, "IM2ROWPOOLSHAPE")

d_R = 2
d_D = 3
d_H = 2
d_W = 2
# Reshape to the output size: 14x14x5x10
out = out.reshape(d_R, d_D, d_H, d_W)
print("out", out)



delta = np.arange(d_R*d_D*d_H*d_W).reshape(d_R,d_D,d_H,d_W) +1
zeros = np.zeros(im2row_pool.shape)
print(zeros)
print(zeros.shape)

zeros[range(x_max_arg.size), x_max_arg] = delta.ravel()
print(zeros)
print(x)

fake_w = np.ones((d_D,d_D,2,2))#2 = kernelsize

dx = row2im_indices(rows = zeros, x_shape = (R * D, 1, H, W), k_size = 2, stride = 2, pad = 0)
print(dx)
