import numpy as np

# MSE stands for Mean Squared Error
# I will mostly be using mse as the loss function because it is fast in both
# implementation and execution
def mse(true_values, predicted_values):
    # mse_loss = ((y_predicted - y_true)^2) / sum(y_true)
    return np.mean(np.power(predicted_values-true_values, 2))

def mseGradient(true_values, predicted_values):
    # mse_gradient = (2 * (y_predicted - y_true)) / len(y_true)
    return 2*(true_values-predicted_values)/np.size(predicted_values)