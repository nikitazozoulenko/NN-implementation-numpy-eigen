import numpy as np
import struct
from scipy.misc import toimage

with open("E:\Datasets/train-images.idx3-ubyte", "rb") as fImages:
    data1 = fImages.read(16)
    data1 = fImages.read(28*28)
with open("E:\Datasets/train-labels.idx1-ubyte", "rb") as fLabels:
    data2 = fLabels.read(8)

for i in range(len(data1)):
    #print(i)
    #print(data1[i])
    pass

print(data1[0])

x = np.array([i for i in data1])
print(x.reshape(1,1,28,28))

toimage(x.reshape(28,28)).show()
