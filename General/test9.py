import numpy as np

def BN(x, gamma, beta):
    mean = np.mean(x, axis = 0)
    variance = np.mean((x-mean)**2, axis = 0)

    xhat = (x-mean) / np.sqrt(variance + epsilon)
    return gamma * xhat + beta

def BN_backwards(x, dy, gamma, beta):
    R = x.shape[0]

    mean = np.mean(x, axis = 0)
    variance = np.mean((x-mean)**2, axis = 0)
    dxdgamma = np.sum((x - mean) / np.sqrt(variance + epsilon) * dy, axis=0)
    dxdbeta = np.sum(dy, axis=0)
    dx1dx0 = gamma / R / np.sqrt(variance + epsilon) * (R * dy - np.sum(dy, axis=0) - (x - mean) / (variance + epsilon) * np.sum(dy * (x - mean), axis=0))

    dJdW[i] = np.empty((2, dxdgamma.shape[0], dxdgamma.shape[1], dxdgamma.shape[2]))
    dJdW[i][0] = dxdgamma
    dJdW[i][1] = dxdbeta
    delta = delta * dx1dx0

    return dJdW[i][0], dJdW[i][1], delta

x = np.arange(2*1*3*3).reshape((2,1,3,3))
print(x)
print(x.shape)
print(np.mean(x,axis = (2,3)))
print(np.mean(x,axis = (2,3)).shape)

print(np.empty(2).shape)


print("test")
y = np.arange(1*2*3*4).reshape((1,2,3,4))
print(y.shape)
print(y.transpose((0,1,3,2)).shape)
