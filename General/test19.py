import numpy as np

first = np.arange(1*1*3*3).reshape(1,1,3,3)*2

second = np.arange(1*1*3*3).reshape(1,1,3,3)*2 +1

x = np.empty((1,2,3,3))
x[0,0] = first
x[0,1] = second
print(x)
R = 1
D = 2
H = 3
W = 3
print(x.reshape(R,D,H*W).transpose(0,2,1).reshape(R,D,H,W))
