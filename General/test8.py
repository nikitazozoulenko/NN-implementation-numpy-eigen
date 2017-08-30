import numpy as np
from statistics import *

x = np.array([1.3,2.2,3.7,4.3,5.1])
np_mean = np.mean(x)
np_var = np.var(x)
print(np_mean)
print(x.sum()/len(x))
print(np_var)
print(sum((x-np_mean)**2)/len(x))
print(sum((x-np_mean)**2)/(len(x)-1))
print(np.var(x, ddof=1))
