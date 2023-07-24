import numpy as np

from .base_loss import Loss


class CrossEntropy(Loss):

    @staticmethod
    def get_loss(neurons, correct):
        """
        CrossEntropy = -sum(correct * ln(neurons)) (pointwise)

        :param neurons: prediction vector
        :param correct: answer vector
        :return: loss
        """
        neurons += 1e-15    # avoid log(0)
        pointwise = np.log(neurons) * correct
        return -np.sum(pointwise)

    def get_gradient(self, neurons, correct):
        """
        ∇CrossEntropy(n, c) = -c/n

        Get the gradient of the last neuron layer
        Returning <∂C_0/∂a_0, ∂C_0/∂a_1, ..., ∂C_0/∂a_n>

        :param neurons: vector of last layer of neural network <a^L_0, a^L_1, ... a^L_n>
        :param correct: vector of ideal y_hat <y_0, y_1, ... y_n>
        :return: vector of partial derivatives
        """
        neurons += 1e-15  # avoid log(0)
        return -correct / neurons
