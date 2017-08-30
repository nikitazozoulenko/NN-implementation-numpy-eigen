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
    print("Prediction after", i, "epochs")
    print("x_last", network.X[len(layers)])
    print("pred", prediction)
    print("Y",Y)
    print("\n", "\n", "\n", "\n", "\n", "\n")

    plt.plot(losses)
    plt.grid(1)
    plt.ylabel("Cost")
    plt.xlabel("iterations")
    plt.show()

def train_network(network, dJdW, learning_rate, mu):
    #MOMENTUM
    for j in range(len(layers)):
        if(network.W[j] is not None):
            network.v[j] = mu * network.v[j] - learning_rate * dJdW[j]
            network.W[j] += network.v[j]



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
mainmean = 0
mainvariance = 0

#settings
batch_size = 8
epochs = 1000
network = CNN(layers = layers, batch_size = batch_size, num_input_channels = image_data.shape[1],
              height = image_data.shape[2], width = image_data.shape[3])
for i in range(epochs):
    #load data into the correct format (4D tensor)
    X = resized_image_data[batch_size*i:batch_size*(i+1), 0:1] / float(255)
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward and backwards pass
    prediction = network.forward(X)
    dJdW = network.backprop(prediction, Y)
    #TESTINGLAYER = 0
    #dJdW_num = dJdW_num = network.compute_numerical_gradient(TESTINGLAYER, Y)
    #print("backprop")
    #print(dJdW[TESTINGLAYER])
    #print("num")
    #print(dJdW_num)
    #print("testing ")
    #print(dJdW_num / dJdW[TESTINGLAYER])
    train_network(network=network, dJdW = dJdW, learning_rate = 0.001, mu = 0.9)

    #Update Graph
    loss = -sum(Y*np.log(prediction+epsilon))
    losses.append(loss.mean())



show_loss(i, prediction, losses)


#mean and variance
if(True):
    for i in range(len(layers)):
        try:
            print(i, network.W[i].shape)
        except AttributeError:
            print("nope")

if(False):
    for i in range(len(layers)):
        if layers[i].function is "convolution":
            network.W_mean[i] = np.mean(network.W[i])
            network.W_variance[i] = np.var(network.W[i])
            print(i)
            print("mean", np.mean(network.W[i]))
            print("var", np.var(network.W[i]))
            print(network.W[i].shape)
            #print(network.W[i])
            print("\n")


if(False):
    for i in range(len(layers)+1):
        if(layers[i].function is not None):
            print("X[",i,"]:", layers[i].function)
            print(network.X[i])
            print()

correct = 0
max_i = 50
#CALC % CORRECT RATE
for i in range(max_i):
    #load data into the correct format (4D tensor)
    X = resized_image_data[batch_size*i:batch_size*(i+1), 0:1] / 255
    Y = np.zeros((batch_size,10,1,1))
    for j in range(batch_size*i, batch_size*(i+1)):
        Y[j-batch_size*i, label_data[j], 0, 0] = 1

    #forward and backwards pass
    prediction = network.forward(X)

    correct += np.sum(np.equal(np.argmax(prediction, axis = 1), np.argmax(Y, axis = 1)))

print("TEST RESULTS:")
print("correct:", correct)
print("antal tester:", max_i*batch_size)
print("correct", 100 * correct/max_i/batch_size, "%")
#prediction = network.forward(X)
#print_matrix(prediction, "pred")
#print_matrix(Y, "Y")
