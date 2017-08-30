import numpy as np
import matplotlib.pyplot as plt
from layer import *
from cnn import *
from math_for_cnn import *
from scipy.misc import imresize, toimage
#from scipy import special, optimize

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

layer_1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 1)

layer_2 = Layer(function = "convolution",
                kernel_size = 2,
                stride = 1,
                pad = 0,
                num_filters = 2)


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
          layer_BN,
          layer_2]



X = np.array([
                [[[1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 1]]],

                [[[0, 1, 1, 1, 0],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0]]]])

Y = np.array([
               [[[1]],
                [[0]]],

               [[[0]],
                [[1]]]])


losses = []


#read data
image_data = np.fromfile("E:\Datasets/train-images.idx3-ubyte", dtype = np.uint8)
image_data = image_data[16:].reshape(60000, 1, 28, 28)
label_data = np.fromfile("E:\Datasets/train-labels.idx1-ubyte", dtype = np.uint8)
label_data = label_data[8:].reshape(60000, 1, 1, 1)
resized_imaged_data = np.empty((60000, 1, 12, 12))

for i in range(3):
    resized_imaged_data[i,0] = imresize(image_data[i,0], (12,12))
    #toimage(resized_imaged_data[i,0]).show()

#used by matplotlib
losses = []

batch_size = X.shape[0]
epochs = 30
showten = False
numbers = np.zeros(40320)
for i in range(epochs):
    network = CNN(layers = layers, batch_size = batch_size, num_input_channels = X.shape[1], height = X.shape[2], width = X.shape[3])
    #forward and backwards pass
    prediction = network.forward(X)
    TESTINGLAYER = 1;
    dJdW_num = network.compute_numerical_gradient(TESTINGLAYER, Y)
    dJdW, remember = network.backprop(prediction, Y, dJdW_num[0])
    numbers += remember
    #delta_num = network.compute_numerical_delta_once_per_forward(TESTINGLAYER, Y)

    print("test start")
    print("backprop")
    print(dJdW[TESTINGLAYER])
    print("num")
    print(dJdW_num)
    print("test done")
    print(dJdW[TESTINGLAYER] / dJdW_num)
    #print("DELTA_NUM \n", delta_num)

    ##MOMENTUM
    #learning_rate = 0.001
    #mu = 0.95
    #for j in range(len(layers)):
    #    if(network.W[j] is not None):
    #        network.v[j] = mu * network.v[j] - learning_rate * dJdW[j]
    #        network.W[j] += network.v[j]

    #Update Graph
    loss = -sum(Y*np.log(prediction+epsilon))
    losses.append(loss.mean())
    if(showten and epochs > 11):
        if(showten and i % int(epochs/4) is 0):
            show_loss(i, prediction, losses)

#show_loss(i, prediction, losses)
print()
print()
print()



correct = 0
max_i = 0
#CALC % CORRECT RATE
for i in range(max_i):
    #forward and backwards pass
    prediction = network.forward(X)

    correct += np.sum(np.equal(np.argmax(prediction, axis = 1), np.argmax(Y, axis = 1)))

#print("TEST RESULTS:")
#print("correct:", correct)
#print("antal tester:", max_i*batch_size)
#print("correct", 100 * correct/max_i/batch_size, "%")
#prediction = network.forward(X)
#print_matrix(prediction, "pred")
#print_matrix(Y, "Y")



print(np.argmin(numbers))
print(numbers[np.argmin(numbers)])

print(numbers[0])
