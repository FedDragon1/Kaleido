import numpy as np

from .paddings import Padding

from ..core import Layer

from ...util import assert_positive_int, assert_str, assert_ndim, assert_valid_stride, assert_valid_kernel_size


class Conv1D(Layer):
    """
    1D convolution layer, temporal convolution.

    This layer creates a convolution kernel that convolve 2d array over
    temporal dimension.
    """

    def __init__(self, kernel_count, kernel_size, stride=1, padding="valid"):
        """
        Creates a 1D convolutional layer.

        :param kernel_count: positive integer, second axis of output
        :param kernel_size: positive integer, the temporal span of kernel
        :param stride: positive integer, steps to skip after each operation
        :param padding: "valid" / "same" / "full"
        """
        self.kernel_size = assert_valid_kernel_size(
            kernel_size,
            1,
            f"{type(self).__qualname__}: kernel size must be positive integer, got {kernel_size}",
        )
        self.kernel_count = assert_positive_int(
            kernel_count,
            f"{type(self).__qualname__}: kernel count must be positive integer, got {kernel_count}",
        )
        self.stride = assert_valid_stride(
            stride, 1, f"{type(self).__qualname__}: kernel size must be positive integer, got {stride}"
        )
        assert_str(
            padding,
            f"{type(self).__qualname__}: padding argument must be string, got {padding} of class {padding.__class__}",
        )
        self.padding = Padding(padding, padding + "1D", self.kernel_size, self.stride)

        super().__init__(None)  # not known until build

    def build_parameters(self, neurons):
        assert_ndim(
            neurons,
            2,
            f"{type(self).__qualname__}: Expected input to be 2D, got {neurons} with {neurons.ndim} dimensions",
        )
        self.input_shape = np.asarray(neurons.shape)

        # (k_size, num_channels, num_filter)
        self.weights = np.random.random(
            (self.kernel_size, self.input_shape[-1], self.kernel_count)
        )
        self.biases = np.zeros((self.kernel_count,))
        self.padding.build(neurons, )

        self.output_shape = np.asarray(self.get_output(neurons).shape)
        self.built = True

    def get_output(self, neurons):
        """
        # TODO: Change documentation

        Convolve the input and return in the shape (output_length, num_kernel)

        One way to think about this approach is to use sliding window analogy.

        (index starts at 1)

              |-neuron[slice]-|
        a[1]  a[2]  a[3]  a[4]  a[5]  a[6]      a[k]
        ⎡a11⎤ ⎡a11⎤ ⎡a21⎤ ⎡a31⎤ ⎡a41⎤ ⎡a51⎤     ⎡ak1⎤
        ⎥a12⎥ ⎥a12⎥ ⎥a22⎥ ⎥a32⎥ ⎥a42⎥ ⎥a52⎥ ... ⎥ak2⎥
        ⎥ ⋮ ⎥ ⎥ ⋮ ⎥ ⎥ ⋮ ⎥ ⎥ ⋮ ⎥ ⎥ ⋮ ⎥ ⎥ ⋮ ⎥     ⎥ ⋮ ⎥
        ⎣a1n⎦ ⎣a1n⎦ ⎣a2n⎦ ⎣a3n⎦ ⎣a4n⎦ ⎣a5n⎦     ⎣akn⎦
        -----------------------------------------------
              ⎡w1⎤  ⎡w1⎤  ⎡w1⎤
              ⎥w2⎥  ⎥w2⎥  ⎥w2⎥
              ⎥⋮ ⎥  ⎥⋮ ⎥  ⎥⋮ ⎥
              ⎣wn⎦  ⎣wn⎦  ⎣wn⎦
        #2    k11   k12   k13                                   ⎡z11⎤
              |---kernel 1---|                                  ⎥z21⎥
        z21 = k11 * a[2] + k12 * a[3] + k13 * a[4] + b[1]  =>   ⎥z31⎥, where L is last possible slide index
                                                                ⎥ ⋮ ⎥
                    ⎡w1⎤  ⎡w1⎤  ⎡w1⎤                            ⎣zL1⎦
                    ⎥w2⎥  ⎥w2⎥  ⎥w2⎥
                    ⎥⋮ ⎥  ⎥⋮ ⎥  ⎥⋮ ⎥
                    ⎣wn⎦  ⎣wn⎦  ⎣wn⎦
                    k11   k12   k13
        #3          |---kernel 1---|
        z31 = k11 * a[3] + k12 * a[4] + k13 * a[5] + b[1] = sum(kernel[1] * a[3:6]) + b[1]
        -----------------------------------------------
                    ⎡w1⎤  ⎡w1⎤  ⎡w1⎤
                    ⎥w2⎥  ⎥w2⎥  ⎥w2⎥
                    ⎥⋮ ⎥  ⎥⋮ ⎥  ⎥⋮ ⎥
                    ⎣wn⎦  ⎣wn⎦  ⎣wn⎦
        #3          k21   k22   k23                             ⎡z01⎤
                    |---kernel 1---|                            ⎥z11⎥
        z32 = k21 * a[3] + k22 * a[4] + k23 * a[5] + b[2]  =>   ⎥z21⎥, where L is last possible slide index
                                                                ⎥ ⋮ ⎥
                          ⎡w1⎤  ⎡w1⎤  ⎡w1⎤                      ⎣zL1⎦
                          ⎥w2⎥  ⎥w2⎥  ⎥w2⎥
                          ⎥⋮ ⎥  ⎥⋮ ⎥  ⎥⋮ ⎥
                          ⎣wn⎦  ⎣wn⎦  ⎣wn⎦
                          k10   k11   k12
        #4                |---kernel 1---|
        z42 = k21 * a[4] + k22 * a[5] + k23 * a[6] + b[2] = sum(kernel[1] * a[4:7]) + b[2]
        -----------------------------------------------
        After this process, the return value is (N = N filters):
            ⎡z11 z12 ... z1N⎤
            ⎥z21 z22 ... z2N⎥
            ⎥ ⋮   ⋮   ⋱   ⋮ ⎥
            ⎣zL1 zL2 ... zLN⎦

        :param neurons: Exactly 2d array that have shape representing (sequence_length (L), #channel (N))
        :return: convolved neurons shape (seq_len, kernel_n)
        """
        neurons = self.padding.pad(neurons)
        x_slices = self.padding.slices()
        self.conv_segments = np.array(
            [neurons[x_slice] for x_slice in x_slices()]
        )

        output = np.einsum("ijk, jkl -> il", self.conv_segments, self.weights) + self.biases
        return output

    def backprop(self, gradients):
        """
        Backpropagation for **this given data**
        Optimizers will handle the batch, this method should only be used internally

        # TODO: Add documentation
        # TODO: Add tests for this

        :param gradients:
        :return:
        """

        # A mimic of dot product for higher order tensors.
        # Notice that `conv_segments` is the Jacobian of output
        # w.r.t one filter of weights. It has to be transposed
        # in order to join calculation. This idea of transposing
        # is captured in einsum.
        nabla_w = np.einsum("ijk, ik -> jk", self.conv_segments, gradients)

        # Notice now `nabla_w` is only one of the Jacobians for one
        # filter. However, all the filters share the same Jacobian.
        # Instead of stacking the same array, create a new axis,
        # as numpy will broadcast the arrays.
        nabla_w = nabla_w[:, :, None]

        nabla_b = np.sum(gradients, axis=0)

        dzda = self.get_dzda_tensor()
        # i -> output length
        # j -> input length
        # k -> input channels
        # l -> kernel_count
        nabla_a = np.einsum("ijkl, il -> jk", dzda, gradients)

        # remove 0 paddings
        nabla_a = self.padding.unpad(nabla_a)

        trainable = {"weights": nabla_w, "biases": nabla_b}
        return trainable, nabla_a

    def get_dzda_tensor(self):
        """
        Generates a 4D array `(output_len, input_len, num_channels, kernel_count)`
        that is used internally to calculate the gradient w.r.t the input neurons.

        :return: 4d array (output_len, input_len, num_channels, kernel_count)
        """
        dzda = np.zeros((self.output_shape[0], *self.input_shape, self.kernel_count))

        for output_n, x_slice in enumerate(self.padding.slices()):
            dzda[output_n, x_slice] = self.weights

        return dzda
