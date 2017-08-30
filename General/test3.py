import numpy as np
import math

def row2im(mat, W, delta_shape, stride = 1, pad = 0):
    #PAD NOT IMPLEMENTED

    x = np.zeros(delta_shape)
    width = delta_shape[3]
    for r in range(mat.shape[0]):
        for d in range(mat.shape[1]):
            for h in range(mat.shape[2]):
                for w in range(mat.shape[3]):
                    error = mat[r, d, h, w]
                    print(x.shape)
                    x[r:r+1, :, h*stride:W.shape[2]+h*stride, w*stride:W.shape[3]+w*stride] += W[d, :, :, :] * error
    #PAD HERE????
    return x

x1 = np.ones((1,32,8,8))
x2 = np.ones((1,32,4,4))
w1 = np.ones((32,32,5,5))

delta = row2im(mat = x2, W = w1, delta_shape = x1.shape, stride = 1, pad = 0)
print(delta)
