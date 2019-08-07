import numpy as np

class NeuralNetwork():
    def __init__(self):
        '''Constructs a neural network with a random seed and random weights.'''
        # seeding for random number generation.
        np.random.seed(1)
        # convert weights to a 3x1 matrix with values from -1 to 1 and a mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        '''Calculates de sigmoid function of x.'''
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        '''Calculate derivate of sigmoid function.'''
        return x * (1 - x)

    def think(self, inputs):
        '''Pass the inputs via the neuron to get output'''
        # convert values to floats
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, inputs, outputs, iters):
        '''Train the network to make accurate predictions while adjusting its weights.'''
        for iter in iters:
            # siphon the training data via the neuron
            out = self.think(inputs)
            # compute error rate for back propagation
            error = outputs - out
            # perform weight adjustments
            adjustments = np.dot(inputs.T, error * self.sigmoid_derivative(out))
            # update weights
            self.synaptic_weights += adjustments