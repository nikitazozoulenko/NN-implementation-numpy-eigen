import numpy as np
import itertools

x = np.arange(2*1*2*2).reshape(2,1,2,2).reshape(2*1*2*2)
print(x)

permutations = np.array(list(itertools.permutations(x)))
for i in range(len(permutations)):
    if (i == 3149 or i == 24385 or i == 315 or i == 29975 or i == 2007 or i ==17134 or i == 20848 or i == 22852 or i == 144 or i == 36325 or i == 80 or i == 2817 or i == 26070 or i == 26581 or i == 24771):
        print(i)
        print(permutations[i].reshape(2,1,2,2))
        pass

y = permutations[315].reshape(2,1,2,2)
