import numpy as np

x = np.array([[[[1,  10, 4],
                [13, 7, 11],
                [2,  11, 5]],

               [[14, 8, 17],
                [3,  12, 6],
                [15, 9, 18]]],

              [[[-1,  -10, -4],
                [-13, -7, -11],
                [-2,  -11, -5]],

                [[-14, -8, -17],
                [-3,  -12, -6],
                [-15, -9, -18]]]])

y = np.array([[[[1,  2, 3],
                [4, 5, 6],
                [7,  8, 9]],

               [[10, 11, 12],
                [13,  14, 15],
                [16, 17, 18]]]])

ori = np.array([[1,-1],
                [10,-10],
                [4,-4],
                [13,-13],
                [7,-7],
                [16,-16],
                [2,-2],
                [11,-11],
                [5,-5],
                [14,-14],
                [8,-8],
                [17,-17],
                [3,-3],
                [12,-12],
                [6,-6],
                [15,-15],
                [9,-9],
                [18,-18]])

num_filters, fD, fH, fW = x.shape

shape = x.shape
#dJdW[i] = np.dot(self.im2rowx[i].T, delta_reshaped).transpose(1,0).reshape(self.W[i].shape) #.T här
#1*2*3*3
original = x.reshape(shape[3]*shape[2]*shape[1], shape[0])
#print(original.reshape(y.shape[3], y.shape[2]*y.shape[1]).T.reshape(y.shape[3],y.shape[1],y.shape[2]).transpose(1,0,2))
#print(original.reshape(3, 6).T.reshape(3,2,3).transpose(1,0,2))




# VIKITG BACKUP print(ori.T.reshape(2,3,6).T.transpose(0,1,2))
print(ori.T.reshape(num_filters,fH,fW*fD).T.reshape(fH,fD,fW,num_filters).transpose(3,1,0,2))
#print(original.T.reshape((num_filters, fD, fW, fH)).transpose(1,0,3,2))

#print(original.reshape(3, 2*6).T.reshape(2,3,2,3).transpose(0,2,1,3))
