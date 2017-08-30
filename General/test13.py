import numpy as np



def BN(x):
    #step 1 mean
    R, D, W, H = x.shape

    mean = np.mean(x, axis = (0,2,3)).reshape((1,D,1,1))
    variance = np.mean((x-mean)**2, axis = (0,2,3)).reshape((1,D,1,1))
    sqrtvariance = np.sqrt(variance + 0.0001)
    xhat = (x-mean)/sqrtvariance
    y = gamma * xhat + beta
    return y

def BN_backwards(x, delta):
    dy = delta
    R, D, W, H = x.shape

    mean = np.mean(x, axis = (0,2,3)).reshape((1,D,1,1))
    variance = np.mean((x-mean)**2, axis = (0,2,3)).reshape((1,D,1,1))
    dxdbeta = np.sum(dy, axis= (0,2,3)).reshape((1,D,1,1))
    dxdgamma = np.sum((x - mean) / np.sqrt(variance) * dy, axis=(0,2,3)).reshape((1,D,1,1))
    dx1dx0 = gamma / R/W/H / np.sqrt(variance) * (R*W*H * dy - np.sum(
        dy, axis=(0,2,3)).reshape((1,D,1,1)) - (x - mean) / variance * np.sum(
        dy * (x - mean), axis=(0,2,3)).reshape((1,D,1,1)))

    return dx1dx0

def other_BN(h):
    epsilon=0.00001
    N = R
    mu = 1/N/H/W*np.sum(h,axis =(0,2,3)).reshape(1,D,1,1)
    sigma2 = 1/N/H/W*np.sum((h-mu)**2,axis=(0,2,3)).reshape(1,D,1,1)
    hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
    y = gamma*hath+beta
    return y

def other_BN_backwards(h, dy):
    eps = 0.00001
    N = R
    mu = 1./N/W/H*np.sum(h, axis = (0,2,3)).reshape(1,D,1,1)
    var = 1./N/W/H*np.sum((h-mu)**2, axis = (0,2,3)).reshape(1,D,1,1)
    dbeta = np.sum(dy, axis=(0,2,3)).reshape(1,D,1,1)
    dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=(0,2,3)).reshape(1,D,1,1)
    dh = (1. / (N*W*H)) * gamma * (var + eps)**(-1. / 2.) * (N*W*H * dy - np.sum(dy, axis=(0,2,3)).reshape(1,D,1,1)
        - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=(0,2,3)).reshape(1,D,1,1))
    return dh
R=1
D=1
H=3
W=3
gamma = np.ones((1,D,1,1))
beta = np.zeros((1,D,1,1))
x = np.arange(R*D*H*W).reshape(R,D,H,W)
ones = np.arange(R*D*H*W).reshape(R,D,H,W) * +1 *2 +1 *3
print(BN(x))
print(other_BN(x))

print(BN_backwards(x, ones))
print(other_BN_backwards(x, ones))

print("TESTING NEW SHIT BELOW")
mean = np.mean(x, axis = (0,2,3))
print(x-mean.reshape(D,1,1))
