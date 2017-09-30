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

ReLU = Layer(function = "ReLU",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

BN = Layer(function = "BN",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

maxpool2_s2 = Layer(function = "maxpool",
                kernel_size = 2,
                stride = 2,
                pad = 0,
                num_filters = None)

conv3_n32_s2 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 2,
                pad = 0,
                num_filters = 32)

conv3_n16_s1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 16)

conv3_n24_s1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 24)

conv3_n32_s1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 32)

conv4_n32_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 32)

conv4_n16_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 16)

conv4_n8_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 8)

conv3_n8_s1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 8)

conv4_n24_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 24)

conv4_n10_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 10)

conv3_n10_s1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 10)

layers =   [
            BN,
            conv3_n16_s1,
            ReLU,
            BN,
            conv3_n16_s1,
            ReLU,
            BN,
            conv3_n16_s1,
            maxpool2_s2,
            ReLU,
            BN,
            conv4_n16_s1,
            ReLU,
            BN,
            conv4_n24_s1,
            ReLU,
            BN,
            conv3_n24_s1,
            ReLU,
            BN,
            conv3_n10_s1
                        ]

E_grad2 = [0]*(len(layers))
E_x2 = [0]*(len(layers))
v = [0]*(len(layers))

def train_network(network, dJdW, learning_rate, mu):
    #ADADELTA
    p = mu
    for j in range(len(layers)):
        if(network.W[j] is not None):
            E_grad2[j] = p * E_grad2[j] + (1-p) * dJdW[j] * dJdW[j]
            RMS_grad = np.sqrt(E_grad2[j] + epsilon)
            RMS_x = np.sqrt(E_x2[j] + epsilon)
            delta_x = - RMS_x / RMS_grad * dJdW[j]
            E_x2[j] = p * E_x2[j] + (1-p) * delta_x * delta_x
            network.W[j] += delta_x

def validate_network():
    correct = 0
    for i in range(int(10000/network.batch_size)):
        #load data into the correct format (4D tensor)
        X = test_images[batch_size*i:batch_size*(i+1), 0:1] / 255
        Y = np.zeros((batch_size,10,1,1))
        for j in range(batch_size*i, batch_size*(i+1)):
            Y[j-batch_size*i, test_lables[j], 0, 0] = 1

        #forward and backwards pass
        prediction = network.forward(X, time = "test")

        #amount of correct answers
        correct += np.sum(np.equal(np.argmax(prediction, axis = 1), np.argmax(Y, axis = 1)))
    validation_accuracy.append(correct/100.0)

#read data
image_data = np.fromfile("E:\Datasets/train-images.idx3-ubyte", dtype = np.uint8)
image_data = image_data[16:].reshape(60000, 1, 28, 28)
label_data = np.fromfile("E:\Datasets/train-labels.idx1-ubyte", dtype = np.uint8)
label_data = label_data[8:].reshape(60000, 1, 1, 1)

test_images = np.fromfile("E:\Datasets/t10k-images.idx3-ubyte", dtype = np.uint8)
test_images = test_images[16:].reshape(10000,1,28,28)
test_lables = np.fromfile("E:\Datasets/t10k-labels.idx1-ubyte", dtype = np.uint8)
test_lables = test_lables[8:].reshape(10000,1,1,1)

#############random_image = np.random.rand(1,1,28,28)

#used by matplotlib
losses = []
validation_accuracy = []

#settings
batch_size = 100
iterations = 600
epochs = 5
network = CNN(layers = layers, batch_size = batch_size, num_input_channels = image_data.shape[1], height = image_data.shape[2], width = image_data.shape[3])
for epoch in range(epochs):
    #shuffle training data
    perm = np.arange(60000)
    np.random.shuffle(perm)
    shuffled_images = np.zeros(image_data.shape)
    shuffled_images[range(perm.size)] = image_data[perm]
    shuffled_lables = np.zeros(label_data.shape, dtype = np.uint8)
    shuffled_lables[range(perm.size)] = label_data[perm]

    #Train loop
    for i in range(iterations):
        #load data into the correct format (4D tensor)
        X = shuffled_images[batch_size*i:batch_size*(i+1), 0:1] / float(255)
        Y = np.zeros((batch_size,10,1,1))
        for j in range(batch_size*i, batch_size*(i+1)):
            Y[j-batch_size*i, shuffled_lables[j], 0, 0] = 1

        #forward pass
        prediction = network.forward(X, time = "train")
        #backwards pass
        dJdW = network.backprop(prediction, Y)
        train_network(network=network, dJdW = dJdW, learning_rate = learning_rate, mu = 0.9)

        #Update Loss Graph
        loss = -sum(Y*np.log(prediction+epsilon))/network.batch_size
        losses.append(loss.mean())

    #check vs validation data and save to pickle
    validate_network()
    network.save_with_pickle(destination = "4x16+2x24_cache_epoch_"+str(epoch)+".p")

print(validation_accuracy)
show_loss(losses)
