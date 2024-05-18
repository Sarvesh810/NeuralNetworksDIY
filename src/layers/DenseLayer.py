import numpy as np

# A Dense layer is connected with all the neurons from the previous layer
class DenseLayer():
    def __init__(self, no_inputs:int, no_neurons: int) -> None:
        # Take for example input array has 3 parameters/data-points and we want this layer to have 5 neurons
        # We will be making a 3x5 matrix here instead of an intuititve 5x3 matrix to help us later witht the dot product
        self.weights = np.random.randn(no_inputs, no_neurons)

        # Each neuron has 1 bias, continuing our example 5 neurons in a layer will have 5 biases
        self.biases = np.random.randn(1, no_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # The activation (value) of each neuron is the sum of (activations*weights) + bias
        # Activation is the output of neuron
        # Weight gives how much importance to give to each neuron in previous layer
        # Biases help in a way de-localizing the function. Assume what 'b' does in equation 'y = mx + b'
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        return self.output

    def backward(self, gradients: np.ndarray, learning_rate: float=0.01) -> np.ndarray:
        
        # It can be divided into 2 steps:
        # 1. Calculating the gradients for inputs, weights and biases
        self.gradients = np.dot(gradients, self.weights.T)
        self.d_weights = np.dot(self.inputs.T, gradients)
        self.d_biases = np.sum(gradients, axis=0, keepdims=True)

        # 2. Updating those weights and biases
        # Learning rate decides how big the steps will be while adjusting the parameters
        self.weights -= learning_rate*self.d_weights
        self.biases -= learning_rate*self.d_biases
        return self.gradients