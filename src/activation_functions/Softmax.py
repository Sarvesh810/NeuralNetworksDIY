import numpy as np

# We are using Softmax Activation layer after the output layer because we want a
# probability distribution of different possible outputs. This will give us a way of comparing
# how correct we are for each sample in the data
class Softmax():
    def forward(self, inputs: np.ndarray):
        # (e^zi)/sum(e^zj) where 'i' is i-th output and j=1...k
        # e -> Euler's number
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponents/np.sum(exponents, axis=1, keepdims=True)

    def backward(self, gradients: np.ndarray) -> None:
        self.gradients = np.empty_like(gradients)
        for i, (out, grad) in enumerate(zip(self.output, gradients)):
            out = out.reshape(-1, 1)
            matrix = np.diagflat(out) - np.dot(out, out.T)
            self.gradients[i] = np.dot(matrix, grad)