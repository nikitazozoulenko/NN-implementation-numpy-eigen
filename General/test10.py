import numpy as np

h = 0.01
x = np.zeros((1,1,3,3))
x1 = np.copy(x)
print(x)
x1[0,0,0,0] += h
print(x)
