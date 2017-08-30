import numpy

class Neural_Network(object): # varför (object)???????
    def __init__(self):
        #Hyperparameters
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        #Weights
        self.W1 = numpy.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        self.z2 = numpy.dot(X, self.W1)
        self.a2 = self.f(self.z2)
        self.z3 = numpy.dot(self.a2, self.W2)
        self.yHat = self.f(self.z3)
        return self.yHat

    def f(self, z):
        #sigmoid function
        return 1/(1+numpy.exp(-z))





X = numpy.matrix(((3, 5), (5, 1), (10, 2)))
y = numpy.matrix(((75),(82),(93)))
Net1 = Neural_Network()
yHat = Net1.forward(X)
print(yHat)
print(y)
