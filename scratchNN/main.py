import NeuralNetwork as NN
import numpy as np

neural_network = NN.NeuralNetwork()

print("Random weights:")
print(neural_network.synaptic_weights)

# training data:
# 4 examples 
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1],
                            [0, 0, 0],
                            [1, 1, 0]])
# corresponding outputs
training_outputs = np.array([[0, 1, 1, 0, 0, 1]]).T

print("The neuron is training...")
# arbitrary number of iterations
neural_network.train(training_inputs, training_outputs, 10000)

print("Adjusted weights after training:")
print(neural_network.synaptic_weights)

while True:
    print("Introduce a new data to process (e.g. '1 0 0'): ", end='')
    try:
        new_data = input().split()
        result = neural_network.think(np.array(new_data))
    except:
        break
    
    print("My best forecast is ", result)
    print()