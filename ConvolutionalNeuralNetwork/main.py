import numpy as np
import matplotlib.pyplot as plt
from layer import *
from cnn import *
from math_for_cnn import *
from scipy.misc import toimage, imresize
#from scipy import special, optimize    #just if i need it in the future, this is kinda of a note for me

#np.set_printoptions(threshold=np.nan)
#np.seterr( over='ignore' )

epsilon = 0.00001

def show_loss(iteration, prediction, losses):
    plt.plot(losses)
    plt.grid(1)
    plt.ylabel("Cost")
    plt.xlabel("iterations")
    plt.show()

layer_conv3_n1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 1)

layer_conv3_n10 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 10)

layer_1 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 2)

layer_2 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 2)

layer_3 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 4)

layer_4 = Layer(function = "convolution",
                kernel_size = 3,
                stride = 1,
                pad = 0,
                num_filters = 4)

layer_tanh = Layer(function = "tanh",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

layer_relu = Layer(function = "ReLU",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)

layer_BN = Layer(function = "BN",
                   kernel_size = None,
                   stride = None,
                   pad = None,
                   num_filters = None)


layers = [layer_1,
        #  layer_relu,
          layer_2,
        #  layer_relu,
          layer_3,
        #  layer_relu,
          layer_4,
        #  layer_relu,
          layer_conv3_n10]

E_grad2 = [0]*(len(layers))
E_x2 = [0]*(len(layers))
v = [0]*(len(layers))

def train_network(network, dJdW, learning_rate, mu):
    #MOMENTUM
    for j in range(len(layers)):
        if(network.W[j] is not None):
            v[j] = mu * v[j] - learning_rate * dJdW[j]
            network.W[j] += v[j]

#read data
image_data = np.fromfile("E:\Datasets/train-images.idx3-ubyte", dtype = np.uint8)
image_data = image_data[16:].reshape(60000, 1, 28, 28)
label_data = np.fromfile("E:\Datasets/train-labels.idx1-ubyte", dtype = np.uint8)
label_data = label_data[8:].reshape(60000, 1, 1, 1)
resized_image_data = np.empty((60000, 1, 11, 11))

for i in range(60000):
    resized_image_data[i,0] = imresize(image_data[i,0], (11,11))
    #toimage(resized_imaged_data[i,0]).show()

#used by matplotlib
losses = []

#settings
batch_size = 4
epochs = 1000
network = CNN(layers = layers, batch_size = batch_size, num_input_channels = image_data.shape[1],
              height = image_data.shape[2], width = image_data.shape[3])
for i in range(epochs):
    #load data into the correct format (4D tensor)
    X = resized_image_data[batch_size*i:batch_size*(i+1), 0:1] / float(255)
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward pass
    prediction = network.forward(X)
    #backwards pass
    dJdW = network.backprop(prediction, Y)
    train_network(network=network, dJdW = dJdW, learning_rate = 0.001, mu = 0.9)

    #Update Graph
    loss = -sum(Y*np.log(prediction+epsilon))
    losses.append(loss.mean())

show_loss(i, prediction, losses)


correct = 0
max_i = 50
#test network predictability rate
for i in range(max_i):
    #load data into the correct format (4D tensor)
    X = resized_image_data[batch_size*i:batch_size*(i+1), 0:1] / 255
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward and backwards pass
    prediction = network.forward(X)

    #amount of correct answers
    correct += np.sum(np.equal(np.argmax(prediction, axis = 1), np.argmax(Y, axis = 1)))

print("TEST RESULTS:")
print("correct:", correct)
print("antal tester:", max_i*batch_size)
print("correct", 100 * correct/max_i/batch_size, "%")
