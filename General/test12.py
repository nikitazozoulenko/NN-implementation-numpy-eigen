import numpy as np

x = np.arange(1*2*3*3).reshape(1,2,3,3) +1
print(x)
x = x.reshape(1,3,3,2).transpose(0,3,1,2)
print(x)
print(x.transpose(0,2,3,1).reshape(1,2,3,3))
