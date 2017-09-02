import numpy as np
from numpy.linalg import inv

x = np.array([[[[5, 1],
   [7, 3]]],


 [[[4, 0],
   [6, 2]]]])
xshape = x.shape
R, D, H, W = xshape
#print(x)
print("x",x)
x0 = np.flip(np.flip(x, axis = 3), axis = 0)
print("x0",x0)
x05 = x0.transpose(0,1,3,2)
print("x0.5",x05)
x1 = x05.reshape(R,D*H*W)
print("x1",x1)
x2 = x1.T
print("x2",x2)
x3 = x2.reshape(R,D,H,W)
print("x3",x3)

full = np.flip(np.flip(x, axis = 3), axis = 0).transpose(0,1,3,2).reshape(R,D*H*W).T.reshape(R,D,H,W)

print("TIME TO DO THE REVERSE:")

y = np.arange(2*1*2*2).reshape(xshape)
print(y)
y1 = y.reshape(D*H*W,R).T.reshape(xshape).transpose(0,1,3,2)
print(y1)

out = np.flip(np.flip(y1, axis = 0), axis = 3)
print(out)

fullout =
