import numpy as np

from .paddings import Padding

from ..core import Layer

from ...util import assert_valid_kernel_size, assert_positive_int, assert_valid_stride, assert_ndim


class Conv2D(Layer):
    def __init__(self, kernel_count, kernel_size, stride=1, padding="valid"):
        self.kernel_size = assert_valid_kernel_size(
            kernel_size,
            f"{type(self).__qualname__}: kernel size must be in format (A, B) where A and B are positive integers, got {kernel_count}",
        )
        self.kernel_count = assert_positive_int(
            kernel_count,
            f"{type(self).__qualname__}: kernel count must be positive integer, got {kernel_count}"
        )
        self.stride = assert_valid_stride(
            stride, 2, f"{type(self).__qualname__}: stride {stride} is not valid."
        )
        self.padding = Padding(padding, padding + "2D")

        super().__init__(None)  # output shape unknown until build

    def build_parameters(self, neurons):
        assert_ndim(
            neurons,
            3,
            f"{type(self).__qualname__}: Expected input to be 3D, got {neurons} with {neurons.ndim} dimensions",
        )
        self.input_shape = np.asarray(neurons.shape)

        # shape = (k_size_x, k_size_y, num_channels, num_filter)
        self.weights = np.random.random(
            (self.kernel_count, *self.kernel_size, self.input_shape[-1], self.kernel_count)
        )

        # shape = (num_filter,)
        self.biases = np.zeros((self.kernel_count,))
        self.padding.build(neurons, self.kernel_size, self.stride)

        self.output_shape = np.asarray(self.get_output(neurons).shape)
        self.built = True

    def get_output(self, neurons):
        neurons = self.padding.pad(neurons)

        

