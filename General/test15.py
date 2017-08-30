import numpy as np
from numpy.linalg import inv

x = np.array([[[[5, 1],
   [7, 3]]],


 [[[4, 0],
   [6, 2]]]])
xshape = x.shape
#print(x)
print(x)
x = np.flip(np.flip(x, axis = 3), axis = 0).transpose(0,1,3,2)
print(x)
print(x.transpose(0,2,3,1).reshape(xshape))
