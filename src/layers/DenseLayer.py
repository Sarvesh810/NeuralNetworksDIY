import numpy as np


class DenseLayer():
    def __init__(self, no_inputs:int, no_neurons: int) -> None:
        # Take for example input array has 3 parameters/data-points and we want this layer to have 5 neurons
        # We wiill be making a 3x5 array here instead of an intuititve 5x3 array to help us later witht the dot product
        self.weights = np.random.randn(no_inputs, no_neurons)

        # Each neuron has 1 bias, continuing our example 5 neurons in a layer will have 5 biases
        self.biases = np.random.randn(no_neurons)

    def foward(self, input: np.ndarray):
        self.output = np.dot(input, self.weights) + self.biases