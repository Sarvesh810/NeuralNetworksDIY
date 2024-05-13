import numpy as np

# A Dense layer is connected with all the neurons from the previous layer
class DenseLayer():
    def __init__(self, no_inputs:int, no_neurons: int) -> None:
        # Take for example input array has 3 parameters/data-points and we want this layer to have 5 neurons
        # We will be making a 3x5 matrix here instead of an intuititve 5x3 matrix to help us later witht the dot product
        self.weights = np.random.randn(no_inputs, no_neurons)

        # Each neuron has 1 bias, continuing our example 5 neurons in a layer will have 5 biases
        self.biases = np.random.randn(no_neurons)

    def forward(self, inputs: np.ndarray) -> None:
        # The activation (value) of each neuron is the sum of (activations*weights) + bias
        # Activation is the output of neuron
        # Weight gives how much importance to give to each neuron in previous layer
        # Biases help in a way de-localizing the function. Assume what 'b' does in equation 'y = mx + b'
        self.output = np.dot(inputs, self.weights) + self.biases