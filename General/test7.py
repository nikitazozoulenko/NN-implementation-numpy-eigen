import numpy as np

x = np.arange(2*5*1*1).reshape((2,5,1,1))
y = np.zeros(2*5*1*1).reshape(2,5,1,1)
y[0,4,0,0] = 1
y[1,4,0,0] = 1

print(y)
print(x)

xargmax = np.argmax(x, axis = 1)
yargmax = np.argmax(y, axis = 1)
print(xargmax)
print(yargmax)

equal = np.equal(xargmax, yargmax)
print(equal)
print(sum(equal))
