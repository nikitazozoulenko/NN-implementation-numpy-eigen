import numpy as np
import matplotlib.pyplot as plt
from layer import *
from cnn import *
from math_for_cnn import *
import PIL
from PIL import Image, ImageFilter, ImageOps
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

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

dream_v = 0
def dream_optimize_step(image, delta):
    #MOMENTUM on image
    global dream_v
    learning_rate = 0.0001
    mu = 0.9
    dream_v = mu * dream_v - learning_rate * delta
    image += dream_v
    return image

original_size = (100,100)
image = np.random.randint(70, size = original_size) / 255
Image.fromarray(image*255).show()

iterations = 20

for _ in range(iterations):
    resized = zoom(image, zoom = (28/original_size[1], 28/original_size[0]), order = 3)
    gradient = network.dream_gradient(image = resized.reshape(1,1,28,28), end = len(layers))
    gradient = gradient.reshape(28,28)
    gradient_zoomed = zoom(gradient, zoom = (original_size[1]/28, original_size[0]/28), order = 3)
    image = dream_optimize_step(image = image, delta = gradient_zoomed)
    np.clip(image, 0, 1)
Image.fromarray(image*255).show()
#    image = dream_optimize_step(image = image, delta = gradient)


#gradient = np.asarray(gradient_image)

#dream_optimize_step(image = image_array, delta = gradient)

#Image.fromarray(image_array*255).show()






# iterations = 5
# for i in range(iterations):
#     image = network.make_step(image = image.reshape(1,1,28,28), end = len(layers))
#     image = image.reshape(28,28)
#     image = np.minimum(1, image)
#     image = np.maximum(0, image)
#     if i % 10 == 0:
#         im = Image.fromarray(image*255)
#         im.show()


#img = ImageOps.fit(img, size, Image.ANTIALIAS)
#def dream(image, layer, )
