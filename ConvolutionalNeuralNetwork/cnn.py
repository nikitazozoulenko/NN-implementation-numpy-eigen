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

        #WEIGHTS
        self.W = [None]*(len(layers))

        #CACHE
        self.im2rowx = [None]*len(layers)
        self.pool_max_arg = [None]*len(layers)
        self.BN_mean = [None]*len(layers)
        self.BN_var = [None]*len(layers)
        self.mean_running_average = [None]*len(layers)
        self.var_running_average = [None]*len(layers)
        self.deltas = [None]*len(layers)

        #init weights/layers
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

                #EWMA initialization
                self.mean_running_average[i] = 0
                self.var_running_average[i] = 0


            elif(layer.function is "maxpool"):
                pad, stride, size= layer.pad, layer.stride, layer.kernel_size
                last_h = int((last_h+pad*2-size)/stride + 1)
                last_w = int((last_w+pad*2-size)/stride + 1)

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def convolution_forward(self, i):
        #variables
        in_R, in_D, in_H, in_W = self.X[i].shape
        num_filters, kernel_D, kernel_H, kernel_W = self.W[i].shape
        out_R = in_R
        out_D = self.W[i].shape[0]
        out_H = int((in_H+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)
        out_W = int((in_W+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)

        #math
        self.im2rowx[i] = im2row(self.X[i], size = self.layers[i].kernel_size, stride = self.layers[i].stride, pad = self.layers[i].pad)
        y = np.dot(self.im2rowx[i], self.W[i].reshape(num_filters, kernel_D*kernel_H*kernel_W).T)
        self.X[i+1] = y.reshape((out_R, out_H, out_W, out_D)).transpose(0, 3, 1, 2)

    def convolution_backprop(self, i, delta, dJdW):
        #variables
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        num_filters, fD, fH, fW = self.W[i].shape

        #derivative w.r.t weights
        delta_reshaped = delta.transpose(0,2,3,1).reshape(delta.shape[0]*delta.shape[2]*delta.shape[3], delta.shape[1])
        dJdW[i] = np.dot(delta_reshaped.T,self.im2rowx[i]).reshape(self.W[i].shape)

        #derivative w.r.t input
        delta2_reshaped = delta.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        W_reshaped = self.W[i].reshape(num_filters, -1)
        rows = np.dot(delta2_reshaped.T, W_reshaped)
        return row2im_indices(rows, self.X[i].shape, k_size, stride, pad)

    def maxpool_forward(self, i):
        #variables
        x = self.X[i]
        R, D, H, W = self.X[i].shape
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        new_width = int((W+pad*2-k_size)/stride + 1)

        #math
        x_reshaped = x.reshape(R * D, 1, H, W)
        im2row_pool = im2row(x_reshaped, size = k_size, stride = stride, pad = pad)
        self.pool_max_arg[i] = np.argmax(im2row_pool, axis = 1)
        self.X[i+1] = im2row_pool[range(self.pool_max_arg[i].size), self.pool_max_arg[i]].reshape(R, D, new_width, new_width)

    def maxpool_backprop(self, i, delta, dJdW):
        #variables
        R, D, H, W = self.X[i].shape
        k_size = self.layers[i].kernel_size
        stride = self.layers[i].stride
        pad = self.layers[i].pad
        new_width = int((W+pad*2-k_size)/stride + 1)

        #derivative w.r.t input
        zeros = np.zeros((R*D*new_width**2, k_size**2))
        zeros[range(self.pool_max_arg[i].size), self.pool_max_arg[i]] = delta.ravel()
        return row2im_indices_maxpool(rows = zeros, x_shape = (R * D, 1, H, W), k_size = k_size, stride = stride, pad = pad).reshape(R,D,H,W)

    def BN_forward_train_time(self, i):
        #variables
        x = self.X[i]
        gamma = self.W[i][0]
        beta = self.W[i][1]
        R, D, W, H = x.shape

        #mean, variance and shift the input
        mean = np.mean(x, axis = (0,2,3)).reshape((1,D,1,1))
        variance = np.mean((x-mean)**2, axis = (0,2,3)).reshape((1,D,1,1))
        xhat = (x-mean) / np.sqrt(variance + epsilon)
        self.X[i+1] = gamma * xhat + beta

        #update cache and exp weighted running average
        alpha = 0.10
        self.BN_mean[i] = mean
        self.BN_var[i] = variance
        self.mean_running_average[i] =  alpha * mean + (1-alpha) * self.mean_running_average[i]
        self.var_running_average[i] =  alpha * variance + (1-alpha) * self.var_running_average[i]

    def BN_forward_test_time(self, i):
        #variables
        x = self.X[i]
        gamma = self.W[i][0]
        beta = self.W[i][1]
        R, D, W, H = x.shape

        #mean, variance and shift the input
        mean = self.mean_running_average[i]
        variance = self.var_running_average[i]
        xhat = (x-mean) / np.sqrt(variance + epsilon)
        self.X[i+1] = gamma * xhat + beta

    def BN_backprop(self, i, delta, dJdW):
        #variables
        h = self.X[i]
        R, D, W, H = h.shape
        gamma = self.W[i][0].reshape(1,D,1,1)
        beta = self.W[i][1]
        mean = self.BN_mean[i]
        var = self.BN_var[i]

        #derivatives
        dgammadx = np.sum((delta - mean) / np.sqrt(var+epsilon) * delta, axis=(0,2,3)).reshape(1,D,1,1)
        dbetadx = np.sum(delta, axis=(0,2,3)).reshape(1,D,1,1)
        new_delta = gamma / (R*W*H) / np.sqrt(var+epsilon) * ((R*W*H) * delta - np.sum(delta, axis=(0,2,3)).reshape(1,D,1,1)
            - (delta - mean) / (var+epsilon) * np.sum(delta * (delta - mean), axis=(0,2,3)).reshape(1,D,1,1))

        dJdW[i] = np.empty(self.W[i].shape)
        dJdW[i][0] = dgammadx
        dJdW[i][1] = dbetadx
        return new_delta

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
        delta = (prediction - actual_value) /self.batch_size

        #loop for the rest of the layers
        for i in range((length-1), -1, -1):
            #self.deltas for the numerical gradients
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

        ##TODO calculating 1 useless thing: dx1dx0, check notes #TODO
        return dJdW

    def compute_numerical_gradient(self, i, Y): #i = the i:th layer,, see my personal cnn model
        if self.W[i] is not None:
            R,D,H,W = self.W[i].shape
            numerical_gradient = np.zeros((R,D,H,W))

            original_w = np.copy(self.W[i])

            prediction = self.forward(self.X[0])
            loss1 = -np.sum(Y*np.log(prediction+epsilon))/self.batch_size

            e = 0.000001
            for r in range(R):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            weights = np.copy(original_w)
                            weights[r,d,h,w] += e
                            self.set_weights(weights, i)
                            prediction = self.forward(self.X[0])
                            loss2 = -np.sum(Y*np.log(prediction+epsilon))/self.batch_size
                            numerical_gradient[r,d,h,w] = (loss2 - loss1) / e

            self.set_weights(original_w, i)
            return numerical_gradient

    def set_weights(self, weights, i):
        self.W[i] = np.copy(weights)

    def compute_numerical_delta_once_per_forward(self, i, Y): #i = the i:th layer, see my personal cnn model    i=2 -> dJdx3
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
