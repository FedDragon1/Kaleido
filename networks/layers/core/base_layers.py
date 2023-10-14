from abc import abstractmethod, ABC

import numpy as np


class Layer(ABC):

    def __init__(self, output_shape, input_shape=None):
        self.built = False
        self.output_shape = np.asarray(output_shape)
        self.input_shape = np.asarray(input_shape) if input_shape is not None else None
        # default to no params
        self.n_param = 0

    def __repr__(self):
        if not self.built:
            return f"<{self.__class__.__qualname__}() {hex(id(self))} (?->{self.output_shape or '?'}) NOT BUILT>"
        return f"<{self.__class__.__qualname__}() {hex(id(self))} " \
               f"({self.input_shape}->{self.output_shape}) {self.n_param} Params>"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, neurons):
        neurons = np.asarray(neurons)
        self.input = neurons
        if not self.built:
            self.build_parameters(neurons)
        self.output = self.get_output(neurons).copy()
        return self.output

    @abstractmethod
    def get_output(self, neurons):
        """Inplace operation is encouraged, since `forward` makes a copy"""
        ...

    @abstractmethod
    def backprop(self, gradients):
        """
        Backpropagation among trainable parameters and input neurons

        :param gradients: gradients from next layer
        :return: (∇trainable, ∇neurons)
        """
        ...

    @abstractmethod
    def build_parameters(self, neurons):
        """
        Build the parameters of this layer.

        Specifically, determines `input_shape` and `n_param` of this layer,
        initialize additional trainable parameters (e.g. weights & biases)

        Lastly, setting the `built` attribute to `True`

        :param neurons: primers
        :return: None
        """
        ...

    def get_trainable(self):
        """
        Get a dict which key is attr name and value is `np.zeros` with corresponding shape
        Default to no trainable parameters

        :return: {attr1: np.zero(att1.shape), attr2: np.zero(attr2.shape), ...}
        """
        return {}

    def input_sanity_check(self, x_in):
        if self.input_shape is None:  # primer
            return
        if self.input_shape != x_in.shape:
            raise TypeError(
                f"Layer {self} of class {self.__class__} expects input shape of {self.input_shape}, got {x_in.shape}.\n"
                f"input: {x_in}"
            )


class PreprocessingLayer(Layer):
    """A dummy for isinstance check. Inherit this class if no meaningful calculation is involved"""

    def build_parameters(self, neurons):
        # No parameters to build, override abstract method
        self.input_shape = neurons.shape

    def backprop(self, gradients):
        """
        Gradient in input layer results in all 1 vector.

        Generally these kind of layers should not be back propagated since it
        should only be used in the very first layers of a network,
        which provides no meaningful calculation.

        Return the original gradients instead, overriding abstract method

        :param gradients: gradients from next layer
        :return: the exact gradients passed in
        """
        return {}, gradients
