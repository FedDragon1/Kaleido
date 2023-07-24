import numpy as np

from networks.layers.core.base_layers import PreprocessingLayer


class Input(PreprocessingLayer):
    """Ordinary Input layer that gives neural network an input shape"""

    def __init__(self, input_shape):
        input_shape = np.asarray(input_shape)
        super().__init__(output_shape=input_shape, input_shape=input_shape)
        self.built = True  # no build

    def __repr__(self):
        return f"<{self.__class__.__qualname__}({self.input_shape}) {hex(id(self))} " \
               f"({self.input_shape}->{self.output_shape}) {self.n_param} Params>"

    def get_output(self, neurons):
        # Do nothing and return
        return neurons
