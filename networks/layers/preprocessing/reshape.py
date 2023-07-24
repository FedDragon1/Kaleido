import numpy as np

from networks.layers.core.base_layers import PreprocessingLayer

from networks.util.assertions import assert_reshape_compatibility, assert_valid_output_shape


class Reshape(PreprocessingLayer):
    """A layer that reshapes input array into desired shape"""

    def __init__(self, output_shape):
        output_shape = assert_valid_output_shape(output_shape)
        super().__init__(output_shape=output_shape)

    def build_parameters(self, neurons):
        self.input_shape = np.asarray(neurons.shape)
        # checks compatible dimension
        assert_reshape_compatibility(
            self.input_shape,
            self.output_shape,
            f"Reshape: cannot reshape array of shape {self.input_shape} ({neurons.size} elements) "
            f"into shape {self.output_shape} ({np.prod(self.output_shape)} elements)",
        )

    def get_output(self, neurons: np.ndarray):
        # reshape the neurons into desired shape
        reshaped = np.reshape(neurons, self.output_shape)
        return reshaped

    def backprop(self, gradients):
        """
        For Reshape layer, it might actually be used as middle layers of a network
        so backpropagation is just to reshape gradients into original input shape

        :param gradients: gradients from next layer
        :return: ({}, ∇neurons)
        """
        gradients = np.reshape(gradients, self.input_shape)
        return {}, gradients
