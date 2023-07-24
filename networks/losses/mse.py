import numpy as np

from .base_loss import Loss


class MSE(Loss):

    @staticmethod
    def get_loss(neurons, correct):
        """
        Get mean squared error given neurons and correct
        MSE = 1/n * sum(neuron - correct)^2

        :param neurons: prediction vector
        :param correct: answer vector
        :return: loss
        """
        squared = (neurons - correct) ** 2
        return np.sum(squared) / len(neurons)

    def get_gradient(self, neurons, correct):
        """
        ∇MSE(n, c) = 2(n - y) / len(n)

        Get the gradient of the last neuron layer
        Returning <∂C_0/∂a_0, ∂C_0/∂a_1, ..., ∂C_0/∂a_n>

        :param neurons: vector of last layer of neural network <a^L_0, a^L_1, ... a^L_n>
        :param correct: vector of ideal y_hat <y_0, y_1, ... y_n>
        :return: vector of partial derivatives
        """
        grad = 2 * (neurons - correct) / len(neurons)
        return grad
