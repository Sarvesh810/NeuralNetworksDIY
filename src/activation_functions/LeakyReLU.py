import numpy as np

# Leaky ReLU solves the problem of dying neurons in ReLU. I believe that if a network
# trains over many epochs, there is a chance of neurons dying due to ReLU's gradient becoming 0
class LeakyReLU():
    def forward(self, inputs: np.ndarray) -> None:
        # Input <= 0 ---> Output = Input*leakage
        # Input > 0 ---> Output = Input
        self.leakage = 0.1
        self.inputs = inputs
        self.output = np.maximum(inputs*self.leakage, inputs)
        return self.output

    def backward(self, gradients: np.ndarray):
        leaky_relu_gradients = np.where(gradients > 0, 1, self.leakage)
        self.gradients = gradients * leaky_relu_gradients
        return self.gradients