import numpy as np
import matplotlib.pyplot as plt
from layer import *
from cnn import *
from math_for_cnn import *
import PIL
from PIL import Image, ImageFilter
from scipy.misc import imresize

np.set_printoptions(threshold=np.nan)
epsilon = 0.00001

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

conv4_n16_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 16)

conv4_n24_s1 = Layer(function = "convolution",
                kernel_size = 4,
                stride = 1,
                pad = 0,
                num_filters = 24)

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

#read data
test_images = np.fromfile("E:\Datasets/t10k-images.idx3-ubyte", dtype = np.uint8)
test_images = test_images[16:].reshape(10000,1,28,28)
test_lables = np.fromfile("E:\Datasets/t10k-labels.idx1-ubyte", dtype = np.uint8)
test_lables = test_lables[8:].reshape(10000,1,1,1)

network = CNN(layers = layers, batch_size = 100, num_input_channels = 1, height = 28, width = 28)
network.load_from_pickle("4x16+2x24_0.9919acc.p")
network.update_batch_size(1)

image = np.random.randint(256, size = (28,28)) / 255
Image.fromarray(image*255).show()

iterations = 5
for i in range(iterations):
    image = network.make_step(image = image.reshape(1,1,28,28), end = len(layers))
    image = image.reshape(28,28)
    image = np.minimum(1, image)
    image = np.maximum(0, image)
    if i % 10 == 0:
        im = Image.fromarray(image*255)
        im.show()
