import numpy as np

def BN(x):
    #step 1 mean
    R, D, W, H = x.shape

    mean = np.mean(x, axis = (2,3)).reshape((R,D,1,1))
    print(mean)
    print(mean.shape)

    print(x-mean)

    #step2: subtract mean vector of every trainings example


    #step3: following the lower branch - calculation denominator

    #step4: calculate variance
    variance = np.mean((x-mean)**2, axis = (2,3)).reshape((R,D,1,1))

    print(variance)

    #step5: add eps for numerical stability, then sqrt
    sqrtvariance = np.sqrt(variance + 0.0001)

    #step6: invert sqrtwar

    #step7: execute normalization
    xhat = (x-mean)/sqrtvariance

    print(xhat)

    #step8: Nor the two transformation steps
    print("\n \n")
    print(gamma)
    print(gamma.shape)

    print(beta)
    print(beta.shape)
    #step9
    y = gamma * xhat + beta

    #y = np.zeros(x.shape)
    #for i in range(x.shape[0]):
    #    y[i] = gamma[i] * xhat[i] + beta[i]

    #store intermediate
    #cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    return y




def BN_backwards(x, dy):
    #backwards
    R, D, W, H = x.shape
    mean = np.mean(x, axis = (2,3)).reshape((R,D,1,1))
    variance = np.mean((x-mean)**2, axis = (2,3)).reshape((R,D,1,1))
    dxdbeta = np.sum(dy, axis= (2,3)).reshape((R,D,1,1))
    dxdgamma = np.sum((x - mean) / np.sqrt(variance) * dy, axis=(2,3)).reshape((R,D,1,1))
    dx1dx0 = gamma / R / np.sqrt(variance) * (R * dy - np.sum(
        dy, axis=(2,3)).reshape((R,D,1,1)) - (x - mean) / variance * np.sum(
        dy * (x - mean), axis=(2,3)).reshape((R,D,1,1)))

    print("beta",dxdbeta.shape)
    return dx1dx0



x = np.arange(2*2*3*3).reshape(2,2,3,3)

R, D, W, H = x.shape
gamma = np.ones((R, D, 1, 1))
beta = np.zeros((R, D, 1, 1))
print(x)
print(x.shape)
print(BN(x))
print(BN_backwards(x,2*x))
