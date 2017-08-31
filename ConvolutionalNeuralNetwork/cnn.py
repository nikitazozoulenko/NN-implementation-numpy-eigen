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
        #WEIGHTS
        self.W = [None]*(len(layers))

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
                pass
                #TODO

    def forward_single_layer(self, i):
        function = self.layers[i].function
        if(function is "convolution"):
            #variables
            in_R, in_D, in_H, in_W = self.X[i].shape
            num_filters, kernel_D, kernel_H, kernel_W = self.W[i].shape
            out_R = in_R
            out_D = self.W[i].shape[0]
            out_H = int((in_H+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)
            out_W = int((in_W+self.layers[i].pad*2-self.layers[i].kernel_size)/self.layers[i].stride + 1)

            #math
            self.im2rowx[i] = im2row(self.X[i], size = self.layers[i].kernel_size, stride = self.layers[i].stride, pad = self.layers[i].pad)
            y = np.dot(self.im2rowx[i], self.W[i].T.reshape(( kernel_D*kernel_H*kernel_W , num_filters )))
            self.X[i+1] = y.reshape((out_R, out_H, out_W, out_D)).transpose(0, 3, 1, 2)

        elif(function is "ReLU"):
            self.X[i+1] = relu(self.X[i])

        elif(function is "tanh"):
            self.X[i+1] = tanh(self.X[i])

        elif(function is "BN"):
            x = self.X[i]
            gamma = self.W[i][0]
            beta = self.W[i][1]
            R, D, W, H = x.shape

            mean = np.mean(x, axis = (2,3)).reshape((R,D,1,1))
            variance = np.mean((x-mean)**2, axis = (2,3)).reshape((R,D,1,1))
            xhat = (x-mean) / np.sqrt(variance + epsilon)

            self.X[i+1] = gamma * xhat + beta

        elif(function is "maxpool"):
            pass
            ######TODO

    def forward(self, input_matrix):
        self.X[0] = input_matrix
        for i in range(len(self.layers)):
            self.forward_single_layer(i)
        return softmax(self.X[len(self.layers)])

    def backprop(self, prediction, actual_value):
        length = len(self.layers)
        dJdW = [None]*(length)

        #hardcoded softmax
        delta = prediction - actual_value

        #loop for the rest of the layers
        for i in range((length-1), -1, -1):
            if(self.layers[i].function is "convolution"):
                #flip 180 degrees
                self.W[i] = self.W[i].transpose((0,1,3,2))

                num_filters, fD, fH, fW = self.W[i].shape
                delta_reshaped = delta.transpose(0,2,3,1).reshape(delta.shape[0]*delta.shape[2]*delta.shape[3], delta.shape[1])
                dJdW[i] = np.dot(self.im2rowx[i].T, delta_reshaped).T.reshape(num_filters,fH,fW*fD).T.reshape(fH,fD,fW,num_filters).transpose(3,1,0,2)
                delta = row2im(mat = delta, W = self.W[i], delta_shape = self.X[i].shape, stride = self.layers[i].stride, pad = self.layers[i].pad)

                #unflip 180 degrees
                self.W[i] = self.W[i].transpose((0,1,3,2))

            elif(self.layers[i].function is "ReLU"):
                delta = delta * relu_prime(self.X[i])

            elif(self.layers[i].function is "tanh"):
                delta = delta * tanh_prime(self.X[i])

            elif(self.layers[i].function is "BN"):
                h = self.X[i]
                delta_shape = delta.shape
                dy = delta#.transpose((0,1,3,2)).transpose(0,2,3,1).reshape(delta_shape)
                gamma = self.W[i][0]
                beta = self.W[i][1]
                R, D, W, H = h.shape

                eps = 0
                N = R
                mu = 1./N/W/H*np.sum(h, axis = (0,2,3)).reshape(1,D,1,1)
                var = 1./N/W/H*np.sum((h-mu)**2, axis = (0,2,3)).reshape(1,D,1,1)
                dbeta = np.sum(dy, axis=(0,2,3)).reshape(1,D,1,1)

                dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=(0,2,3)).reshape(1,D,1,1)
                dh = (1. / N /W /H) * gamma * (var + eps)**(-1. / 2.) * (N*W*H * dy - np.sum(dy, axis=(0,2,3)).reshape(1,D,1,1)
                    - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=(0,2,3)).reshape(1,D,1,1))

                #DELTA IS PROBABLY SHAPED WRONG ||||||OR PROBABLY NOT
                dJdW[i] = np.empty(self.W[i].shape)

                dJdW[i][0] = dgamma
                dJdW[i][1] = dbeta

                delta = dh

            elif(self.layers[i].function is "maxpool"):
                pass
                ######TODO

            ##TODO calculating 1 useless thing, check notes #TODO

            ##TESTING BATCH SIZE DIVISION
            dJdW[i] /= self.batch_size

        return dJdW

    def compute_numerical_gradient(self, i, Y): #i = the i:th layer,, see my personal cnn model
        if self.W[i] is not None:
            R,D,H,W = self.W[i].shape
            numerical_gradient = np.zeros((R,D,H,W))

            original_w = np.copy(self.W[i])

            prediction = self.forward(self.X[0])
            loss1 = -np.sum(Y*np.log(prediction+epsilon))

            e = 0.0001
            for r in range(R):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            weights = np.copy(original_w)
                            weights[r,d,h,w] += e
                            self.set_weights(weights, i)
                            prediction = self.forward(self.X[0])
                            loss2 = -np.sum(Y*np.log(prediction+epsilon))
                            numerical_gradient[r,d,h,w] = (loss2 - loss1) / e

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
            self.forward_single_layer(j)
        prediction1 = softmax(self.X[len(self.layers)])
        loss1 = -np.sum(Y*np.log(prediction1+epsilon))

        e = 0.0001
        for r in range(R):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        #+h for each variable
                        self.X[i+1][r,d,h,w] +=e

                        #loss2
                        for j in range(i+1, len(self.layers)):
                            self.forward_single_layer(j)
                        prediction2 = softmax(self.X[len(self.layers)])
                        loss2 = -np.sum(Y*np.log(prediction2+epsilon))

                        #grad
                        numerical_gradient[r,d,h,w] = (loss2 - loss1) / e

                        #undo +h
                        self.X[i+1][r,d,h,w] -=e

        return numerical_gradient
