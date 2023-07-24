import numpy as np

from .base_activation import Activation


__all__ = ("Tanh", "Sigmoid")


class Sigmoid(Activation):

    def get_output(self, neurons):
        return 1 / (1 + np.exp(-neurons))

    def get_gradient(self, gradient):
        """
        a = sigmoid(z)
        ∂a/∂z = sigmoid'(z), where sigmoid'(z) -> sigmoid(z) * (1 - sigmoid(z))

        ∂a/∂z * ∂C/∂a = sigmoid'(z) * gradient     (in the sense of backpropagation)

        :param gradient: ∂C/∂a, "cumulative" gradient of the next layer
        :return: ∂C/∂z
        """
        sigmoid_d = self.output * (1 - self.output)
        gradient = sigmoid_d * gradient
        return {}, gradient


class Tanh(Activation):

    def get_output(self, neurons):
        return np.tanh(neurons)

    def get_gradient(self, gradient):
        """
        a = tanh(z)
        ∂a/∂z = tanh'(z), where tanh'(z) -> 1 - tanh²(z)

        ∂a/∂z * ∂C/∂a = tanh'(z) * gradient     (in the sense of backpropagation)

        :param gradient: ∂C/∂a, "cumulative" gradient of the next layer
        :return: ∂C/∂z
        """
        tanh_d = 1 - self.output ** 2
        gradient = tanh_d * gradient
        return {}, gradient
