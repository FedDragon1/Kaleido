from abc import abstractmethod

import numpy as np

from ..util import SubclassDispatcherMeta


class Loss(metaclass=SubclassDispatcherMeta):

    def forward(self, neurons, correct):
        self.neurons = np.asarray(neurons)
        self.correct = np.asarray(correct)
        self.loss = self.get_loss(neurons, correct)
        return self.loss

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

    def backprop(self):
        """
        Get the derivative of activation given neuron values.
        Default behavior does not consider parameters in activations.
        Override this method if trainable parameters exist
        """
        return self.get_gradient(self.neurons, self.correct)

    @staticmethod
    @abstractmethod
    def get_loss(neurons, correct):
        """
        Get the scaler value loss given prediction `neurons` and expected `correct`

        :param neurons: prediction vector
        :param correct: answer vector
        :return: loss
        """
        ...

    @abstractmethod
    def get_gradient(self, neurons, correct):
        """
        Get the gradient of the last neuron layer

        Returning <∂C_0/∂a_0, ∂C_0/∂a_1, ..., ∂C_0/∂a_n>

        :param neurons: vector of last layer of neural network <a^L_0, a^L_1, ... a^L_n>
        :param correct: vector of ideal y_hat <y_0, y_1, ... y_n>
        :return: vector of partial derivatives
        """
        ...
