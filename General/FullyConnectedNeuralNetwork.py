import numpy as np
import matplotlib.pyplot as plt
from scipy import special, optimize

def print_matrix(data, name):
    print(name)
    print(data)
    print("\n")

#rectified linear units
def relu(x):
    return x * (x > 0)

#derivative of relu
def relu_prime(x):
    return 1 * (x > 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def nothing(x):
    return x

def nothing_prime(x):
    return 1




class Layer(object):
    def __init__(self, neurons = 1, activation_fun = relu, activation_fun_prime = relu_prime):
        self.neurons = neurons
        self.activation_fun = activation_fun
        self.activation_fun_prime = activation_fun_prime






class ProductCreator(object):

    def __init__(self, batch_size = 1):
        self.batch_size = batch_size
        self.X = np.array((batch_size, 2))
        self.Y = np.array((batch_size, 1))

    def gen_values(self):
        self.X = np.random.random((batch_size, 2))
        self.Y = self.X[:,0:1] * self.X[:,1:2]







class NeuralNetwork(object):
    #CONSTRUCTOR
    def __init__(self, layers, batch_size = 1):
        self.batch_size = batch_size
        self.layers = layers
        self.learning_rate = 0.01
        self.W = [None]*(len(layers)-1)
        self.Z = [None]*(len(layers)-1)
        self.A = [None]*(len(layers)-1)
        self.E_grad2 = [None]*(len(layers)-1)
        self.E_x2 = [None]*(len(layers)-1)

        #Init weights with Xavier Initialization + other init
        for i in range(len(layers)-1):
            if(layers[i].activation_fun is relu):
                self.W[i] = np.random.randn(layers[i].neurons, layers[i+1].neurons) / np.sqrt(layers[i].neurons / 2*2)
            else:
                self.W[i] = np.random.randn(layers[i].neurons, layers[i+1].neurons) / np.sqrt(layers[i].neurons)
            self.E_grad2[i] = np.zeros((layers[i].neurons, layers[i+1].neurons))
            self.E_x2[i] = np.zeros((layers[i].neurons, layers[i+1].neurons))
            #self.v[i] = np.zeros((layers[i].neurons, layers[i+1].neurons))

    #FORWARD PROPAGATION
    def forward(self, X, Y):
        #first layer
        self.Z[0] = np.dot(X, self.W[0])
        self.A[0] = layers[1].activation_fun(self.Z[0])

        #rest of the loop
        for i in range(1, len(layers)-1):
            self.Z[i] = np.dot(self.A[i-1], self.W[i])
            self.A[i] = layers[i+1].activation_fun(self.Z[i])

        return self.A[len(layers)-2]

    def getParams(self):
        params = np.concatenate((self.W[0].ravel(), self.W[1].ravel()))
        return params

    def setParams(self, params):
        W0_start = 0;
        W0_end = self.layers[0].neurons*self.layers[1].neurons
        self.W[0] = np.reshape(params[W0_start:W0_end],
                                (self.layers[0].neurons, self.layers[1].neurons))
        W1_end = W0_end + self.layers[1].neurons*self.layers[2].neurons
        self.W[1] = np.reshape(params[W0_end:W1_end],
                                (self.layers[1].neurons, self.layers[2].neurons))

    def computeNumericalGradient(self, X, Y):
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 0.0001

        for p in range(len(paramsInitial)):
            perturb[p] = e
            self.setParams(paramsInitial+perturb)
            loss2 = 0.5*sum((Y - self.forward(X, Y))**2)

            self.setParams(paramsInitial - perturb)
            loss1 = 0.5*sum((Y - self.forward(X, Y))**2)

            numgrad[p] = (loss2-loss1) / (2*e)

            perturb[p] = 0

        self.setParams(paramsInitial)

        return numgrad


    #BACKPROPAGATION
    def backprop(self, prediction, X, Y):
        dJdW = [None]*(len(layers)-1)
        #calculate the gradient

        #delta = -(Y-prediction)
        #dJdW[1] = np.dot(np.transpose(self.A[0]), delta)

        #delta = np.dot(delta, np.transpose(self.W[1])) * relu_prime(self.Z[0])
        #dJdW[0] = np.dot(np.transpose(X), delta)






        k = len(layers)-2

        delta = -(Y-prediction) * self.layers[k+1].activation_fun_prime(self.Z[k])
        dJdW[k] = np.dot(np.transpose(self.A[k-1]), delta)

        for i in range(len(layers)-3, 0, -1):
            delta = np.dot(delta, np.transpose(self.W[i+1]))
            delta = delta * self.layers[i+1].activation_fun_prime(self.Z[i])
            dJdW[i] = np.dot(np.transpose(self.A[i-1]), delta)

        delta = np.dot(delta, np.transpose(self.W[1]))
        delta = delta * self.layers[1].activation_fun_prime(self.Z[0])
        dJdW[0] = np.dot(np.transpose(X), delta)

        return dJdW





inputlayer = Layer(neurons = 2,
                   activation_fun = nothing,
                   activation_fun_prime = nothing_prime)
hidden_1 = Layer(neurons = 8,
                   activation_fun = relu,
                   activation_fun_prime = relu_prime)
hidden_2 = Layer(neurons = 8,
                   activation_fun = relu,
                   activation_fun_prime = relu_prime)
hidden_3 = Layer(neurons = 8,
                   activation_fun = relu,
                   activation_fun_prime = relu_prime)
hidden_4 = Layer(neurons = 8,
                   activation_fun = relu,
                   activation_fun_prime = relu_prime)
hidden_5 = Layer(neurons = 8,
                  activation_fun = relu,
                  activation_fun_prime = relu_prime)
outputlayer = Layer(neurons = 1,
                   activation_fun = nothing,
                   activation_fun_prime = nothing_prime)

layers = [inputlayer,
          hidden_1,
          hidden_2,
          hidden_3,
          hidden_4,
          hidden_5,
          outputlayer]

batch_size = 4
network = NeuralNetwork(layers, batch_size = batch_size)
creator = ProductCreator(batch_size = batch_size)

data = np.array([[1.0, 1.0, 1.0],
                 [1.0, 0.5, 0.5],
                 [0.5, 1.0, 0.5],
                 [0.5, 0.5, 0.25]])
Y = data[:, 2:3]
X = data[:,0:2]

losses = []

for i in range(10001):
    creator.gen_values()
    X = creator.X
    Y = creator.Y
    prediction = network.forward(X, Y)
    loss = 0.5*sum((Y - prediction)**2)
    losses.append(loss.mean())
    dJdW = network.backprop(prediction, X, Y)
    #dJdWnum = network.computeNumericalGradient(X, Y)

    #MOMENTUM
    p = 0.95
    epsilon = 0.00001
    for j in range(len(network.layers)-1):
        network.E_grad2[j] = p * network.E_grad2[j] + (1-p) * dJdW[j] * dJdW[j]
        RMS_grad = np.sqrt(network.E_grad2[j] + epsilon)
        RMS_x = np.sqrt(network.E_x2[j] + epsilon)
        delta_x = - RMS_x / RMS_grad * dJdW[j]
        network.E_x2[j] = p * network.E_x2[j] + (1-p) * delta_x * delta_x
        network.W[j] += delta_x

plt.plot(losses)
plt.grid(1)
plt.ylabel("Cost")
plt.xlabel("iterations")
plt.show()

prediction = network.forward(X, Y)
print_matrix(prediction, "pred")
print_matrix(Y, "Y")








##MOMENTUM
#mu = 0.95
#for j in range(len(network.layers)-1):
#    network.v[j] = mu * network.v[j] - network.learning_rate * dJdW[j]
#    network.W[j] += network.v[j]
