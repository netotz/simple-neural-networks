import scratchNN.NeuralNetwork as NN
import numpy as np

neural_network = NN.NeuralNetwork()

print("Random weights:")
print(neural_network.synaptic_weights)

# training data:
# 4 examples 
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

print("The network is training...")
# arbitrary number of iterations
neural_network.train(training_inputs, training_outputs, 15000)

print("Adjusted weights after training:")
print(neural_network.synaptic_weights)

print("Introduce a new data to process (e.g. '1 0 0'): ", end='')
new_data = input().split()
result = neural_network.think(np.array(new_data))
print("My best forecast is ", )