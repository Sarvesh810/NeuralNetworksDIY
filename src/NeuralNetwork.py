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
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    # Backwward mehtod for the whole network
    def backward(self, gradients: np.ndarray, learning_rate: float):
        for layer in self.layers[::-1]:
            layer.backward(gradients)
            gradients = layer.gradients

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, batch_size: int, learning_rate: float=0.01):
        # The data is shuffled and fed into the model quite a few times
        # This helps in better learning of the model
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            # Loop over the data batch-wise and train the model
            for batch in range(0, X_train, batch_size):
                pass
        # Training porcess can be divided into 3 steps:
        # 1. Forward Step
