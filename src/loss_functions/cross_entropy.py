import numpy as np

# Cross-Entropy loss works well with Softmax activation
# It is a probability-based loss function that. For higher divergences from the true values,
# the loss is exponentially higher as it utilises log function in its calculation of loss.
class CrossEntropy():
    def forward(self, true_values: np.ndarray, predicted_values: np.ndarray):
        return -np.mean(np.sum(true_values * np.log(predicted_values), axis=1))
    
    def backward(self, true_values: np.ndarray, predicted_values: np.ndarray):
        return predicted_values - true_values