class Layer(object): #maxpool, ReLU, convolution, tanh, BN
    def __init__(self, function = "convolution", kernel_size = 3, stride = 1, pad = 0, num_filters = 1 ):
        self.function = function
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.num_filters = num_filters
