import numpy as np

# We are using Softmax Activation layer after the output layer because we want a
# probability distribution of different possible outputs. This will give us a way of comparing
# how correct we are for each sample in the data
class Softmax():
    def forward(self, inputs: np.ndarray):
        # (e^zi)/sum(e^zj) where 'i' is i-th output and j=1...k
        # e -> Euler's number
        exponents = np.exp(inputs - np.max(inputs))
        self.output = exponents/np.sum(exponents)

    def backward(self, gradients: np.ndarray):
        matrix = np.diag(self.output) - np.outer(self.output, self.output)
        self.gradients = np.dot(gradients, matrix)