import numpy as np

from .paddings import Padding

from ..core import Layer

from ...util import assert_positive_int, assert_str, assert_ndim, assert_valid_stride, assert_valid_kernel_size, \
    requires_layer_build


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
        Convolve the input and return in the shape (output_length, num_kernel)

        Using the sliding window strategy, first obtain all the segments to be convolved
        which have the same shape as weights as a 3d array S.
        S.shape -> (output_length, kernel_size, input_channels) -> ijk

        <latex name="S"></latex>

        The `weights` array is also a 3d array W.
        `W.shape` $\rightarrow$ `(kernel_size, input_channels, kernel_count)` `(jkl)`

        <latex name="W"></latex>

        The output array is a 2d array Z.
        `Z.shape` $\rightarrow$ `(output_length, kernels_count)` `(il)`

        Thus, the calculation of Z can be captured with this signature:

        $${\rm ijk}, {\rm jkl} \rightarrow {\rm il}$$

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

    @requires_layer_build
    def backprop(self, gradients):
        r"""
        Backpropagation for **this given data**
        Optimizers will handle the batch, this method should only be used internally

        ### Gradient of Output w.r.t Loss

        The parameter `gradients` passed from next layer represents the gradient $\frac{\partial C}{\partial z}$

        $$\frac{\partial C}{\partial z} = \frac{\partial C}{\partial z}=\begin{bmatrix}
        z_{1, 1} & z_{1, 2} & \cdots & z_{1, l} \\
        z_{2, 1} & z_{2, 2} & \cdots & z_{2, l} \\
        \vdots & \vdots & \ddots & \vdots \\
        z_{i, 1} & z_{i, 2} & \cdots & z_{i, l} \\
        \end{bmatrix}$$

        where $i$ stands for output length and $l$ stands for kernel count.

        > Notice that this is a 2D tensor `(il)` with shape `(output_length, kernel_count)`

        ### Gradient w.r.t Weights

        To calculate the gradient of loss w.r.t the weight, we need to find $\frac{\partial z}{\partial w}$.

        $$\frac{\partial z}{\partial w}=\begin{bmatrix}
        a_1 & a_2 & \cdots & a_j \\
        a_{1+s} & a_{2+s} & \cdots & a_{j+s} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{1+is} & a_{2+is} & \cdots & a_{j+is}
        \end{bmatrix}$$

        where $s$ stands for stride, $i$ stands for output length, and $j$ stands for kernel size.

        > Notice that this is a 3D tensor `(ijk)`, with shape `(output_length, kernel_size, input_channels)`,
        > and it is exactly `self.conv_segments`.

        Then, by multiplying $\frac{\partial z}{\partial w}$ and $\frac{\partial C}{\partial z}$ together,
        results in a tensor with same shape as weight `(jkl)` `(kernel_size, input_channels, output_channel)`.

        $$\rm ijk,il\rightarrow jkl$$

        Thus, $\frac{\partial C}{\partial w}$ can be evaluated with the following code:
        ```np.einsum("ijk, il -> jkl", self.conv_segments, gradients)```

        ### Gradient w.r.t Biases

        For each channel there is a single bias associated with it. Thus, the shape of bias is `(l)` `(kernel_count)`.

        The gradient, $\frac{\partial C}{\partial z}$, has a shape `(il)` `(output_length, kernel_count)`.

        Then the gradient of loss w.r.t the bias can be calculated as $\rm il\rightarrow l$. Effectively the same as
        summing all the elements along the axis of `output_length`.

        Thus, $\frac{\partial C}{\partial b}$ can be calculated with the following code:
        ```np.sum(gradients, axis=0)```

        ### Gradient w.r.t Input Neurons

        To calculate the gradient of loss w.r.t the input, we need to find $\frac{\partial z}{\partial a^{(p)}}$
        (Padded input neuron).

        This tensor is computed by method `get_dzda_tensor`. The return value of this method is a 4D tensor `(ijkl)`
        with shape `(output_length, padded_input_length, input_channels, num_filters)`

        > Notice that `padded_input_length` must be used. Since the convolution operation is performed based on
        > the padded tensor not the input tensor.

        Then, by multiplying $\frac{\partial z}{\partial a^{(p)}}$ and $\frac{\partial C}{\partial z}$ together,
        results in a tensor with same shape as weight `(jk)` `(padded_input_length, input_channels)`.

        $$\rm ijkl,il\rightarrow jk$$

        Thus, $\frac{\partial C}{\partial a^{(p)}$ can be evaluated with the following code:
        ```np.einsum("ijkl, il -> jk", self.get_dzda_tensor(), gradients)```

        Lastly, unpad $\frac{\partial C}{\partial a^{(p)}}$ to get $\frac{\partial C}{\partial a}$

        :param gradients: $\frac{\partial C}{\partial z}$
        :return: None
        """

        # A mimic of dot product for higher order tensors.
        # Notice that `conv_segments` is the Jacobian of output
        # w.r.t one filter of weights. It has to be transposed
        # in order to join calculation. This idea of transposing
        # is captured in einsum.
        nabla_w = np.einsum("ijk, il -> jkl", self.conv_segments, gradients)

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

    @requires_layer_build
    def get_dzda_tensor(self):
        """
        Generates a 4D array `(output_len, pad_input_len, num_channels, kernel_count)`
        that is used internally to calculate the gradient w.r.t the input neurons.

        :return: 4d array (output_len, input_len, num_channels, kernel_count)
        """
        dzda = np.zeros((self.output_shape[0], *self.padding.processed_shape, self.kernel_count))

        x_slices = self.padding.slices()

        for output_n, x_slice in enumerate(x_slices()):
            dzda[output_n, x_slice] = self.weights

        return dzda
