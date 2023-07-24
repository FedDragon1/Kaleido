from abc import abstractmethod

import numpy as np

from networks.layers import Layer

from networks.util import SubclassDispatcherMeta


class Activation(Layer, metaclass=SubclassDispatcherMeta):

    def __init__(self):
        super().__init__(None)

    def backprop(self, gradient):
        """
        Get the derivative of activation given neuron values.
        Default behavior does not consider parameters in activations.
        Override this method if trainable parameters exist

        :param gradient: neuron values
        :return: derivative at given neurons
        """
        gradient = np.asarray(gradient)
        return self.get_gradient(gradient)

    @abstractmethod
    def get_gradient(self, neurons):
        ...

    def build_parameters(self, neurons):
        self.input_shape = self.output_shape = np.asarray(neurons.shape)
        self.built = True
