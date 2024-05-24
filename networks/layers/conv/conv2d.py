import numpy as np

from .paddings import Padding

from ..core import Layer

from ...util import assert_valid_kernel_size, assert_positive_int, assert_valid_stride, assert_ndim, \
    requires_layer_build


class Conv2D(Layer):
    def __init__(self, kernel_count, kernel_size, stride=1, padding="valid"):
        self.kernel_size = assert_valid_kernel_size(
            kernel_size,
            2,
            f"{type(self).__qualname__}: kernel size must be in format (A, B) where A and B are positive integers, got {kernel_count}",
        )
        self.kernel_count = assert_positive_int(
            kernel_count,
            f"{type(self).__qualname__}: kernel count must be positive integer, got {kernel_count}"
        )
        self.stride = assert_valid_stride(
            stride, 2, f"{type(self).__qualname__}: stride {stride} is not valid."
        )
        self.padding = Padding(padding, padding + "2D", self.kernel_size, self.stride)

        super().__init__(None)  # output shape unknown until build

    def build_parameters(self, neurons):
        assert_ndim(
            neurons,
            3,
            f"{type(self).__qualname__}: Expected input to be 3D, got {neurons} with {neurons.ndim} dimensions",
        )
        self.input_shape = np.asarray(neurons.shape)

        # shape = (k_size_y, k_size_x, num_channels, num_filter)
        self.weights = np.random.random(
            (*self.kernel_size, self.input_shape[-1], self.kernel_count)
        )

        # shape = (num_filter,)
        self.biases = np.zeros((self.kernel_count,))
        self.padding.build(neurons)

        self.output_shape = np.asarray(self.get_output(neurons).shape)
        self.built = True

    def get_output(self, neurons):
        neurons = self.padding.pad(neurons)
        x_slices, y_slices = self.padding.slices()
        self.conv_segments = np.array(
            [
                [neurons[y_slice, x_slice] for x_slice in x_slices()]
                for y_slice in y_slices()
            ]
        )

        # i -> output_length_y
        # j -> output_length_x
        # k -> kernel_size_x
        # l -> kernel_size_y
        # m -> input_channels
        # n -> output_channels
        output = np.einsum("ijklm, klmn -> ijn", self.conv_segments, self.weights) + self.biases
        return output

    @requires_layer_build
    def backprop(self, gradients):

        nabla_w = np.einsum("ijklm, ijn -> klmn", self.conv_segments, gradients)

        nabla_b = np.sum(gradients, axis=(0, 1))

        # TODO: This tensor consumes excessive amount of memory...
        dzda = self.get_dzda_tensor()
        # i -> output_length_y
        # j -> output_length_x
        # k -> padded_input_length_y
        # l -> padded_input_length_x
        # m -> input_channels
        # n -> kernel_count
        nabla_a = np.einsum("ijklmn, ijn -> klm", dzda, gradients)

        nabla_a = self.padding.unpad(nabla_a)

        # the gradient is really huge depending on the input size...
        # regularize the gradient by making the biggest element -1 or 1
        nabla_w /= np.max(np.abs(nabla_w))
        nabla_b /= np.max(np.abs(nabla_b))

        trainable = {"weights": nabla_w, "biases": nabla_b}
        return trainable, nabla_a

    @requires_layer_build
    def get_dzda_tensor(self):
        """
        Generates a 6D array
        `(output_len_y, output_len_y, pad_input_length_y, pad_input_length_x, num_channels, kernel_count)`
        that is used internally to calculate the gradient w.r.t the input neurons.

        :return: 6d array (output_len, input_len, num_channels, kernel_count)
        """
        dzda = np.zeros((*self.output_shape[:-1], *self.padding.processed_shape, self.kernel_count))

        x_slices, y_slices = self.padding.slices()

        for output_y_n, y_slice in enumerate(y_slices()):
            for output_x_n, x_slice in enumerate(x_slices()):
                dzda[output_y_n, output_x_n, y_slice, x_slice] = self.weights

        return dzda

