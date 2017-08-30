import numpy as np

def im2row(mat, size = 3, stride = 1, pad = 0):
    #only 4D tensors
    assert mat.ndim == 4

    #tensor characteristics - Width Height Depth    R = fourth dimension
    R, D, H, W = mat.shape

    #square image
    assert W == H

    #x is the padded tensor
    x = np.zeros((R, D, H+pad*2, W+pad*2))
    x[:, :, pad:H+pad, pad:W+pad] = mat

    #find the dim of the new layer
    W_new_float = (W+pad*2-size)/stride + 1
    W_new = int(W_new_float)
    assert W_new == W_new_float

    #im2row
    y = np.empty((R*W_new**2, D*size**2))
    for i in range(R):
        for j in range(W_new):
            for k in range(W_new):
                y[i*W_new*W_new + j*W_new + k] = x[i, :, j:size+j, k:size+k].ravel()

    return y

def row2im(mat, W, delta_shape, stride = 1, pad = 0):
    #PAD NOT IMPLEMENTED

    new_R, new_D, new_H, new_W = delta_shape
    x = np.zeros(delta_shape)
    width = delta_shape[3]
    for r in range(mat.shape[0]):
        for d in range(mat.shape[1]):
            for h in range(mat.shape[2]):
                for w in range(mat.shape[3]):
                    error = mat[r, d, h, w]
                    x[r:r+1, :, h*stride:W.shape[2]+h*stride, w*stride:W.shape[3]+w*stride] += W[d:d+1, :, :, :] * error

    x = x.transpose(0,2,3,1).reshape(delta_shape)

    #PAD HERE????
    return x

def print_matrix(data, name):
    print(name)
    print(data)
    print("\n")

#rectified linear units
def relu(x):
    return x * (x > 0)

#derivative of relu
def relu_prime(x):
    return 1 * (x > 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def nothing(x):
    return x

def nothing_prime(x):
    return 1

def softmax(X):
    exp = np.exp(X)
    exp_sum = np.sum(exp, axis = 1)
    for i in range(exp.shape[0]):
        exp[i] = exp[i] / (exp_sum[i] + 0.00001)
    return exp