import numpy as np

R,D,H,W = 1, 1, 3, 3
x = np.arange(R*D*H*W).reshape(R,D,H,W)
print(x)
x.transpose(0,1,3,2).reshape(R,D,H*W).transpose(0,2,1).reshape(R,D,H,W)
print(x)

y = np.arange(5*2*3*3).reshape(5,2,3,3)
print("y", y)
print("y.T", y.T)
print("real y reshaped", y.transpose())

test = slice(None), 1, 2, 3
print(test)

print(np.arange(4).reshape(1, 1, 2, 2))
w = np.arange(4).reshape(1, 1*2*2).T
print(w)
