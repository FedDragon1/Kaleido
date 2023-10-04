import numpy as np

from .paddings import Padding

from ..core import Layer

from ...util import assert_positive_int, assert_str, assert_ndim, assert_valid_stride, assert_valid_kernel_size


class Conv1D(Layer):
    def __init__(self, kernel_count, kernel_size, stride=1, padding="valid"):
        self.kernel_size = assert_valid_kernel_size(
            kernel_size,
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
        self.padding = Padding(padding, padding + "1D")

        super().__init__(None)  # not known until build

    def build_parameters(self, neurons):
        assert_ndim(
            neurons,
            2,
            f"{type(self).__qualname__}: Expected input to be 2D, got {neurons} with {neurons.ndim} dimensions",
        )
        self.input_shape = np.asarray(neurons.shape)

        # TODO: Change this to (k_size, num_channels, num_filter)
        # shape = (num_filter, k_size, num_channels)
        self.weights = np.random.random(
            (self.kernel_count, self.kernel_size, self.input_shape[-1])
        )
        self.biases = np.zeros((self.kernel_count,))
        self.padding.build(neurons, self.kernel_size, self.stride)

        self.output_shape = np.asarray(self.get_output(neurons).shape)
        self.built = True

    def get_output(self, neurons):
        """
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
        neurons = self.padding.process(neurons)
        self.conv_segments = np.array(
            [neurons[_slice] for _slice in self.padding.slices()]
        )
        output = np.array(
            [
                np.sum(_slice * self.weights, axis=(1, 2)) + self.biases
                for _slice in self.conv_segments
            ]
        )
        return output

    def backprop(self, gradients):
        """
        Backpropagation for **this given data**
        Optimizers will handle the batch, this method should only be used internally

        keymap:
            i -> output index
            f -> kernel (filter) index
            r -> input row
            c -> input channel
            j -> weight row
            k -> weight column
            s -> conv_segments
            L -> output sequence length
            N -> num kernel
            F -> num filter
            s[i].shape == w[f].shape

        target function (index starts at 1):
            z[i][f] = sum(w[f] * s[i]) + b[f]

        weights:
            ∂z[i][f]/∂w[f][j][k]
                  (only look at row `j` since all other rows does not contain this variable targeted)
                = ∂[w[f][j][1] * s[i][j][1] + w[f][j][2] * s[i][j][2]
                        + ... + w[f][j][k] * s[i][j][k] + ... + b[f]] / ∂w[f][j][k]
                = s[i][j][k]
            ∂z[:, f]/∂w[f][j][k]
                  (differentiate for all z[1][f], z[2][f], ... z[L][f])
                  ⎡s[1][j][k]⎤
                = ⎥s[2][j][k]⎥
                  ⎥     ⋮    ⎥
                  ⎣s[L][j][k]⎦
            ∂C/∂w[f][j][k] = ∂C/∂z * ∂z/∂w[f][j][k]
                             (other filters are not relevant to this specific weight)
                           = ∂C/∂z[:, f] · ∂z[:, f]/∂w[f][j][k]
                           = grad[:, f] · s[:, j, k]
            ∂C/∂w[f] = ∂C/∂z * ∂z/∂w[f]
                     = ⎡grad[:, f]·s[:, j, k]  grad[:, f]·s[:, j, k]  ...  grad[:, f]·s[:, j, k]⎤
                       ⎥grad[:, f]·s[:, j, k]  grad[:, f]·s[:, j, k]  ...  grad[:, f]·s[:, j, k]⎥
                       ⎥          ⋮                      ⋮             ⋱             ⋮          ⎥
                       ⎣grad[:, f]·s[:, j, k]  grad[:, f]·s[:, j, k]  ...  grad[:, f]·s[:, j, k]⎦
                       (for broadcasting, create 2 new axes and take transpose)
                     = sum(∂C/∂z[:, f][None, None].T * ∂z[:, f]/∂w[f], axis=0)
                     = sum(grad[:, f][None, None].T * s, axis=0)
            ∂C/∂w = sum(grad[:, f][None, None].T * s, axis=0) for f in N

        biases:
            ∂z[i][f]/∂b[f]
                = ∂[w[f][0][1] * s[i][0][1] + w[f][0][2] * s[i][0][2]
                        + ... + w[f][j][k] * s[i][j][k] + ... + b[f]] / ∂b[f]
                = 1
            ∂z[:, f]/∂b[f] = <1, 1, 1, 1, 1, ..., 1> len L
            ∂C/∂b[f] = ∂C/∂z[:, f] · ∂z[:, f]/∂b[f]
                     = sum(grad[:, f])
            ∂C/∂b = sum(grad, axis=0)

        inputs:
            ∂z/∂a[r][c]
                  (slide the reversed weight transpose across a zero matrix from top to bottom)
                  (the dimension of zero matrix is (no_stride_length, F), where)
                  (no_stride_length = L + stride * (L - 1), then take only the rows mod stride equals 0)
                = w.T[r, ::-1, :]
                = ⎡w[1][1][1] w[2][1][1] ... w[F][1][1]⎤
                  ⎥     0          0     ...      0    ⎥
                  ⎥     ⋮          ⋮      ⋱       ⋮    ⎥
                  ⎣     0          0     ...      0    ⎦
            ∂C/∂a[r][c] = sum(∂C/∂z * ∂z/∂a[r][c])
            ∂C/∂a = sum(∂C/∂z * ∂z/∂a, axis=(2, 3))
                  = sum(grad * ∂z/∂a, axis=(2, 3))

        :param gradients: gradients from next layer (usually local gradient from activation fn)
        :return: ({"weights": ∇weights, "biases": ∇biases},  ∇neurons)
        """
        nabla_w = np.array(
            [
                np.sum(gradients[:, f][None, None].T * self.conv_segments, axis=0)
                for f in range(self.kernel_count)
            ]
        )

        nabla_b = np.sum(gradients, axis=0)

        no_stride_length = self.output_shape[0] + (self.stride - 1) * (self.output_shape[0] - 1)
        dzda = np.zeros((*self.input_shape, *self.output_shape))
        # generate the matrices
        for nrow in range(self.input_shape[0]):
            for ncol in range(self.input_shape[1]):
                rotated_weights = self.weights.T[ncol, ::-1]
                matrix = np.zeros((no_stride_length, self.kernel_count))
                # put weights on zero matrix, shift down by nrow columns
                start_index = max(0, nrow - len(rotated_weights) + 1)
                end_index = nrow + 1
                weights_fit = end_index - start_index
                matrix[start_index: end_index] = rotated_weights[-weights_fit:]
                # take only stride compatible elements
                matrix = matrix[::self.stride]
                print(matrix)
                dzda[nrow][ncol] = matrix
        nabla_a = np.sum(gradients * dzda, axis=(2, 3))

        trainable = {"weights": nabla_w, "biases": nabla_b}
        return trainable, nabla_a
