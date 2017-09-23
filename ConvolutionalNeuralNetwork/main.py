import numpy as np
import matplotlib.pyplot as plt
from layer import *
from cnn import *
from math_for_cnn import *
#from scipy import special, optimize    #just if i need it in the future, this is kinda of a note for me

np.set_printoptions(threshold=np.nan)
#np.seterr( over='ignore' )

epsilon = 0.00001

def show_loss(losses):
    plt.plot(losses)
    plt.grid(1)
    plt.ylabel("Cost")
    plt.xlabel("iterations")
    plt.show()

tanh = Layer(function = "tanh",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

relu = Layer(function = "ReLU",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

BN = Layer(function = "BN",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

conv_3_n1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 1)

conv_3_n2 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 2)

conv_3_n4 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 4)

conv_3_n8 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 8)

conv_3_n16 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 16)

conv_3_n10 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 10)

conv_4_n8 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 8)

conv_2_n10 = Layer(function = "convolution",
                kernel_size = 2,
                stride = 1,
                pad = 0,
                num_filters = 10)

conv_2_n1 = Layer(function = "convolution",
                kernel_size = 2,
                stride = 1,
                pad = 0,
                num_filters = 1)

maxpool_k2s2 = Layer(function = "maxpool",
                kernel_size = 2,
                stride = 2,
                pad = 0,
                num_filters = None)

layers =   [
            maxpool_k2s2,
            BN,
            conv_3_n2,
            conv_3_n2,
            maxpool_k2s2,
            BN,
            conv_3_n4,
            BN,
            conv_3_n10]

E_grad2 = [0]*(len(layers))
E_x2 = [0]*(len(layers))
v = [0]*(len(layers))

def train_network(network, dJdW, learning_rate, mu):
    #ADADELTA
    p = 0.95
    for j in range(len(layers)):
        if(network.W[j] is not None):
            E_grad2[j] = p * E_grad2[j] + (1-p) * dJdW[j] * dJdW[j]
            RMS_grad = np.sqrt(E_grad2[j] + epsilon)
            RMS_x = np.sqrt(E_x2[j] + epsilon)
            delta_x = - RMS_x / RMS_grad * dJdW[j]
            E_x2[j] = p * E_x2[j] + (1-p) * delta_x * delta_x
            network.W[j] += delta_x

#read data
image_data = np.fromfile("E:\Datasets/train-images.idx3-ubyte", dtype = np.uint8)
image_data = image_data[16:].reshape(60000, 1, 28, 28)
label_data = np.fromfile("E:\Datasets/train-labels.idx1-ubyte", dtype = np.uint8)
label_data = label_data[8:].reshape(60000, 1, 1, 1)

#used by matplotlib
losses = []

#settings
batch_size = 8
epochs = 400
network = CNN(layers = layers, batch_size = batch_size, num_input_channels = image_data.shape[1], height = image_data.shape[2], width = image_data.shape[3])

for i in range(epochs):
    #load data into the correct format (4D tensor)
    X = image_data[batch_size*i:batch_size*(i+1), 0:1] / float(255)
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward pass
    prediction = network.forward(X)
    #backwards pass
    dJdW = network.backprop(prediction, Y)
    train_network(network=network, dJdW = dJdW, learning_rate = 0.005, mu = 0.9)

    #Update Graph
    loss = -sum(Y*np.log(prediction+epsilon))/network.batch_size
    losses.append(loss.mean())



show_loss(losses)


correct = 0
max_i = 40
#test network predictability rate
for i in range(max_i):
    #load data into the correct format (4D tensor)
    X = image_data[batch_size*i:batch_size*(i+1), 0:1] / 255
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward and backwards pass
    prediction = network.forward(X)


    #amount of correct answers
    correct += np.sum(np.equal(np.argmax(prediction, axis = 1), np.argmax(Y, axis = 1)))

print("PREDICTION", prediction)
print("Y", Y)

print("TEST RESULTS:")
print("correct:", correct)
print("antal tester:", max_i*batch_size)
print("correct", 100 * correct/max_i/batch_size, "%")
