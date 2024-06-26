import numpy as np

# I will be using Rectified Linear Unit (ReLU) as the activation function for the hidden layers
# Why ReLU? Because it doesn't have the problem of vanishing gradient and it's derivative/gradient
# is very easy and fast to compute. Although vanishing gradient problem usually occurs in
# deep neural nets which probably would not be the case here
class ReLU():
    def forward(self, inputs: np.ndarray) -> None:
        # Input <= 0 ---> Output = 0
        # IInput > 0 ---> Output = Input
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, gradients: np.ndarray):
        relu_gradients = np.where(self.inputs > 0, 1, 0)
        self.gradients = gradients * relu_gradients
        return self.gradients