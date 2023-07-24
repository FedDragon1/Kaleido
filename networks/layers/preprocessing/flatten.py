import numpy as np

from networks.layers.core.base_layers import PreprocessingLayer


class Flatten(PreprocessingLayer):

    def __init__(self):
        # shape unknown until build
        super().__init__(None)

    def build_parameters(self, neurons):
        self.input_shape = np.asarray(neurons.shape)
        self.output_shape = np.asarray(neurons.flatten().shape)

    def get_output(self, neurons):
        output = np.ravel(neurons)  # in-place
        return output

    def backprop(self, gradients):
        """
        Flatten layer is essentially the same as the reshape layer,
        so their logic are the same

        :param gradients: gradients from next layer
        :return: ({}, ∇neurons)
        """
        gradients = np.reshape(gradients, self.input_shape)
        return {}, gradients
