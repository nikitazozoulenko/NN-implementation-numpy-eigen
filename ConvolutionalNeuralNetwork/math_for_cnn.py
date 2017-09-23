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
                y[i*W_new*W_new + j*W_new + k] = x[i, :, j*stride:size+j*stride, k*stride:size+k*stride].ravel()

    return y

def row2im_indices(rows, x_shape, k_size=3, stride=1, pad = 0):
    x = np.zeros(x_shape)
    R, D, H, W = x_shape
    width_range = int((W+pad*2-k_size)/stride + 1)
    for r in range(R):
        for h in range(width_range):
            for w in range(width_range):
                x[r:r+1, :, h*stride:k_size+h*stride, w*stride:k_size+w*stride] += rows[R*(h*width_range+w)+r].reshape(1,D,k_size,k_size)
    return x

def row2im_indices_maxpool(rows, x_shape, k_size=3, stride=1, pad = 0):
    x = np.zeros(x_shape)
    R, D, H, W = x_shape
    width_range = int((W+pad*2-k_size)/stride + 1)
    for r in range(R):
        for h in range(width_range):
            for w in range(width_range):
                x[r:r+1, :, h*stride:k_size+h*stride, w*stride:k_size+w*stride] += rows[r*width_range**2+h*width_range+w].reshape(1,D,k_size,k_size)
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
