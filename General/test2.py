import numpy as np

delta = np.array([[[[1,2,3],
                [4,5,6],
                [7,8,9]],

              [[10,11,12],
                [13,14,15],
                [16,17,18]],

              [[19,20,21],
                [22,23,24],
                [25,26,27]]],

                              [[[10,20,30],
                                [40,50,60],
                                [70,80,90]],

                              [[100,110,120],
                                [130,140,150],
                                [160,170,180]],

                              [[190,200,210],
                                [220,230,240],
                                [250,260,270]]]])

print(delta)
print(delta.shape)

delta_reshaped = delta.transpose(0,2,3,1).reshape(delta.shape[0]*delta.shape[2]*delta.shape[3], delta.shape[1]) #eller transpose(0,3,2,1)

print(delta_reshaped)
print(delta_reshaped.shape)
