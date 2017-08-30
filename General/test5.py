import numpy as np
import struct
from scipy.misc import toimage

image_data = np.fromfile("E:\Datasets/train-images.idx3-ubyte", dtype = np.uint8)
image_data = image_data[16:].reshape(60000, 1, 28, 28)
label_data = np.fromfile("E:\Datasets/train-labels.idx1-ubyte", dtype = np.uint8)
label_data = label_data[8:].reshape(60000, 1, 1, 1)

#toimage(image_data[0, 0]).show()

y = np.zeros((1,10,1,1))

i = 0
y[0, label_data[i], 0, 0] = 1

print(y)
