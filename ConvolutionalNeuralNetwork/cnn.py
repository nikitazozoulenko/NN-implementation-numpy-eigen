import numpy as np
from layer import *
from math_for_cnn import *
import itertools

epsilon = 0.00001

class CNN(object):

    def __init__(self, layers, batch_size = 32, num_input_channels = 3, height = 28, width = 28):
        self.layers=layers
        self.batch_size = batch_size

        #INPUT
        self.X = [None]*(len(layers)+1)
        self.im2rowx = [None]*len(layers)
        self.pool_max_arg = [None]*len(layers)
        #WEIGHTS
        self.W = [None]*(len(layers))

        self.deltas = [None]*(len(layers))

        last_n = num_input_channels
        last_h = height
        last_w = width
        for i in range(len(layers)):
            layer = self.layers[i]
            if(layer.function is "convolution"):
                n, d, size = layer.num_filters, last_n, layer.kernel_size
                pad, stride = layer.pad, layer.stride
                #attempt at xavier initialization
                self.W[i] = np.random.randn(n, d, size, size) / np.sqrt(d*size*size)
                #when done:
                last_n = n
                last_h = int((last_h+pad*2-size)/stride + 1)
                last_w = int((last_w+pad*2-size)/stride + 1)

            elif(layer.function is "BN"):
                #init gamma to one, beta to zeros
                #subscript zero = gamma, subscript one = beta
                self.W[i] = np.empty((2, last_n, 1, 1))
                self.W[i][0] = np.ones((last_n, 1, 1))
                self.W[i][1] = np.zeros((last_n, 1, 1))

            elif(layer.function is "maxpool"):
                last_n = n
                last_h = int((last_h+pad*2-size)/stride + 1)
                last_w = int((last_w+pad*2-size)/stride + 1)

    def convolution_forward(self, i):
        # #variables
        # in_R, in_D, in_H, in_W = self.X[i].shape
        # num_filters, kernel_D, kernel_H, kernel_W = self.W[i].shape
        # out_R = in_R
        # out_D = self.W[i].shape[0]
        # out_H = int((in_H+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)
        # out_W = int((in_W+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)
        #
        # #math
        # self.im2rowx[i] = im2row(self.X[i], size = self.layers[i].kernel_size, stride = self.layers[i].stride, pad = self.layers[i].pad)
        # y = np.dot(self.im2rowx[i], self.W[i].T.reshape(( kernel_D*kernel_H*kernel_W , num_filters )))
        # self.X[i+1] = y.reshape((out_R, out_H, out_W, out_D)).transpose(0, 3, 1, 2)

        n_filters = self.W[i].shape[0]
        self.im2rowx[i] = im2row(self.X[i], size = self.layers[i].kernel_size, stride = self.layers[i].stride, pad = self.layers[i].pad)
        W_col = self.W[i].reshape(n_filters, -1)

        in_R, in_D, in_H, in_W = self.X[i].shape

        n_x = self.X[i].shape[0]
        h_out = int((in_H+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)
        w_out = int((in_W+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)

        out = W_col @ self.im2rowx[i].T
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.X[i+1] = out

    def convolution_backprop(self, i, delta, dJdW):
        # #flip 180 degrees
        # self.W[i] = self.W[i].transpose((0,1,3,2))
        #
        # num_filters, fD, fH, fW = self.W[i].shape
        # delta_reshaped = delta.transpose(0,2,3,1).reshape(delta.shape[0]*delta.shape[2]*delta.shape[3], delta.shape[1])
        # dJdW[i] = np.dot(self.im2rowx[i].T, delta_reshaped).T.reshape(num_filters,fH,fW*fD).T.reshape(fH,fD,fW,num_filters).transpose(3,1,0,2)
        # delta = row2im(mat = delta, W = self.W[i], delta_shape = self.X[i].shape, stride = self.layers[i].stride, pad = self.layers[i].pad)
        #
        # #unflip 180 degrees
        # self.W[i] = self.W[i].transpose((0,1,3,2))
        #
        # return delta

        R, D, H, W = self.X[i].shape
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        new_width = int((W+pad*2-k_size)/stride + 1)

        n_filter, d_filter, h_filter, w_filter = self.W[i].shape
        num_filters, fD, fH, fW = self.W[i].shape
        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dJdW[i] = np.dot(delta_reshaped, self.im2rowx[i]).reshape(self.W[i].shape)

        dR, dD, dH, dW = delta.shape
        #delta = delta.transpose(0,1,3,2).reshape(dR,dD,dH*dW).transpose(0,2,1).reshape(dR,dD,dH,dW)


        W_reshape = self.W[i].reshape(n_filter, -1)
        dx_rows = (W_reshape.T @ delta_reshaped).T
        delta = row2im_indices(dx_rows, self.X[i].shape, k_size, stride=stride, pad = pad)
        return delta
        # R, D, H, W = self.X[i].shape
        # num_filters = self.W[i].shape[0]
        # k_size = self.layers[i].kernel_size
        # stride = self.layers[i].stride
        # pad = self.layers[i].pad
        # new_width = int((W+pad*2-k_size)/stride + 1)
        #
        # delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        #
        # w_reshaped = self.W[i].reshape(D*k_size**2, num_filters)
        # dx_rows = np.dot(self.im2rowx[i], w_reshaped)
        # delta = row2im_indices(dx_rows, self.X[i].shape, k_size, stride, pad)
        # return delta

    def maxpool_forward(self, i):
        x = self.X[i]
        R, D, H, W = self.X[i].shape
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        new_width = int((W+pad*2-k_size)/stride + 1)

        x_reshaped = x.reshape(R * D, 1, H, W)
        im2row_pool = im2row(x_reshaped, size = k_size, stride = stride, pad = pad)
        self.pool_max_arg[i] = np.argmax(im2row_pool, axis = 1)

        self.X[i+1] = im2row_pool[range(self.pool_max_arg[i].size), self.pool_max_arg[i]].reshape(R, D, new_width, new_width)

    def maxpool_backprop(self, i, delta, dJdW):
        R, D, H, W = self.X[i].shape
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        new_width = int((W+pad*2-k_size)/stride + 1)

        zeros = np.zeros((R*D*new_width**2, k_size**2))
        zeros[range(self.pool_max_arg[i].size), self.pool_max_arg[i]] = delta.ravel()

        return row2im_indices(rows = zeros, x_shape = (R * D, 1, H, W), k_size = k_size, stride = stride, pad = pad).reshape(R,D,H,W)

    def BN_forward_train_time(self, i):
        x = self.X[i]
        gamma = self.W[i][0]
        beta = self.W[i][1]
        R, D, W, H = x.shape

        mean = np.mean(x, axis = (0,2,3)).reshape((1,D,1,1))
        variance = np.mean((x-mean)**2, axis = (0,2,3)).reshape((1,D,1,1))
        xhat = (x-mean) / np.sqrt(variance + epsilon)

        self.X[i+1] = gamma * xhat + beta

    def BN_forward_test_time(self, i):
        #TODO
        pass

    def BN_backprop(self, i, delta, dJdW):
        h = self.X[i]
        R, D, H, W = self.X[i].shape

        gamma = self.W[i][0]
        beta = self.W[i][1]

        gamma = gamma.reshape(1,D,1,1)

        eps = 0.000001
        mu = 1./R/H/W*np.sum(h, axis = (0,2,3)).reshape(1,D,1,1)
        var = 1./R/H/W*np.sum((h-mu)**2, axis = (0,2,3)).reshape(1,D,1,1)
        dbeta = np.sum(delta, axis=(0,2,3)).reshape(1,D,1,1)
        dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * delta, axis=(0,2,3)).reshape(1,D,1,1)
        dh = (1. / R /H /W) * gamma * (var + eps)**(-1. / 2.) * (R*H*W * delta - np.sum(delta, axis=(0,2,3)).reshape(1,D,1,1)
           - (h - mu) * (var + eps)**(-1.0) * np.sum(delta * (h - mu), axis=(0,2,3)).reshape(1,D,1,1))

        dJdW[i] = np.zeros(self.W[i].shape)

        dJdW[i][0] = dgamma
        dJdW[i][1] = dbeta

        return dh

    def forward_single_layer(self, i, BN_function):
        function = self.layers[i].function
        if(function is "convolution"):
            self.convolution_forward(i)

        elif(function is "maxpool"):
            self.maxpool_forward(i)

        elif(function is "BN"):
            BN_function(i)

        elif(function is "ReLU"):
            self.X[i+1] = relu(self.X[i])

        elif(function is "tanh"):
            self.X[i+1] = tanh(self.X[i])
        #print(self.X[i+1].shape)

    def forward(self, input_matrix):
        self.X[0] = input_matrix
        for i in range(len(self.layers)):
            self.forward_single_layer(i, BN_function = self.BN_forward_train_time)
        return softmax(self.X[len(self.layers)])

    def forward_test_time(self, input_matrix):
        self.X[0] = input_matrix
        for i in range(len(self.layers)):
            self.forward_single_layer(i, BN_function = self.BN_forward_test_time)
        return softmax(self.X[len(self.layers)])

    def backprop(self, prediction, actual_value):
        length = len(self.layers)
        dJdW = [None]*(length)

        #hardcoded softmax
        delta = (1./self.batch_size) * (prediction - actual_value)

        #loop for the rest of the layers
        for i in range((length-1), -1, -1):
            self.deltas[i] = delta
            if(self.layers[i].function is "convolution"):
                delta = self.convolution_backprop(i,delta,dJdW)

            elif(self.layers[i].function is "BN"):
                delta = self.BN_backprop(i,delta,dJdW)

            elif(self.layers[i].function is "maxpool"):
                delta = self.maxpool_backprop(i,delta,dJdW)

            elif(self.layers[i].function is "ReLU"):
                delta = delta * relu_prime(self.X[i])

            elif(self.layers[i].function is "tanh"):
                delta = delta * tanh_prime(self.X[i])

        ##TODO calculating 1 useless thing, check notes #TODO
        return dJdW

    def compute_numerical_gradient(self, i, Y): #i = the i:th layer,, see my personal cnn model
        if self.W[i] is not None:
            R,D,H,W = self.W[i].shape
            numerical_gradient = np.zeros((R,D,H,W))

            original_w = np.copy(self.W[i])

            e = 0.000001
            for r in range(R):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            weights = np.copy(original_w)
                            weights[r,d,h,w] += e
                            self.set_weights(weights, i)
                            prediction = self.forward(self.X[0])
                            loss1 = -np.sum(Y*np.log(prediction+epsilon))/self.batch_size

                            weights[r,d,h,w] -= 2*e
                            self.set_weights(weights, i)
                            prediction = self.forward(self.X[0])
                            loss2 = -np.sum(Y*np.log(prediction+epsilon))/self.batch_size
                            numerical_gradient[r,d,h,w] = (loss1 - loss2) / (2*e)

            self.set_weights(original_w, i)
            return numerical_gradient

    def set_weights(self, weights, i):
        self.W[i] = np.copy(weights)

    def compute_numerical_delta_once_per_forward(self, i, Y): #i = the i:th layer,, see my personal cnn model
        #i=2    ->   dJdx3
        R,D,H,W = self.X[i+1].shape
        numerical_gradient = np.zeros((R,D,H,W))
        #loss1
        for j in range(i+1, len(self.layers)):
            self.forward_single_layer(j, BN_function = self.BN_forward_train_time)
        prediction1 = softmax(self.X[len(self.layers)])
        loss1 = -np.sum(Y*np.log(prediction1+epsilon))/self.batch_size

        e = 0.0001
        for r in range(R):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        #+h for each variable
                        self.X[i+1][r,d,h,w] +=e

                        #loss2
                        for j in range(i+1, len(self.layers)):
                            self.forward_single_layer(j, BN_function = self.BN_forward_train_time)
                        prediction2 = softmax(self.X[len(self.layers)])
                        loss2 = -np.sum(Y*np.log(prediction2+epsilon))/self.batch_size

                        #grad
                        numerical_gradient[r,d,h,w] = (loss2 - loss1) / e

                        #undo +h
                        self.X[i+1][r,d,h,w] -=e

        return numerical_gradient
