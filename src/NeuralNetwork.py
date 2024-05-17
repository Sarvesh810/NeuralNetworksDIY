import numpy as np
from layers.DenseLayer import DenseLayer
from src.activation_functions.ReLU import ReLU
from src.activation_functions.Softmax import Softmax
from src.layers.DenseLayer import DenseLayer
from src.loss_functions.cross_entropy import CrossEntropy

class NeuralNetwork():
    def __init__(self) -> None:
        self.layers = []
        self.loss_function = CrossEntropy()

    def add(self, layer) -> None:
        self.layers.append(layer)

    # Forward method for the whole network
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Go over every layer in the model
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    # Backward method for the whole network
    def backward(self, gradients: np.ndarray, learning_rate: float) -> None:
        for layer in self.layers[::-1]:
            layer.backward(gradients)
            gradients = layer.gradients

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, batch_size: int, learning_rate: float=0.01) -> None:
        # An epoch is complete when the neural network gas gone over the data once. We repeat this
        # process of the neural network going over the dataset many times, shuffling the data
        # each time to make it learn more from the same data
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])       # Similar to python's range() method
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            # Now, for each epoch, divide the data into batches
            # Each batch is fed as a single input to the network
            # This speeds up the learning process with a tradeoff with some accuracy
            for batch in range(0, X_train.shape[0], batch_size):
                X = X_train[batch:batch+batch_size]     # Slice the data in a batch iterarting over batches
                Y = Y_train[batch:batch+batch_size]
                
                # Now, the training process can be divided into 2 steps:
                # Step 1: Forward Propagation and calculate loss
                output = self.forward(inputs=X)
                loss = self.loss_function.forward(true_values=Y, predicted_values=output)

                # Step 2: Calculate gradients w.r.t loss and Backward Propagation
                gradients = self.loss_function.backward(true_values=Y, predicted_values=output)
                self.backward(gradients=gradients)

