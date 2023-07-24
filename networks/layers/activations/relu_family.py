import numpy as np

from .base_activation import Activation


__all__ = ("ReLU", "LeakyReLU")


class ReLU(Activation):

    def get_output(self, neurons):
        """
                 /
        0   ____/
                0

        Max between each element and 0

        :param neurons: input
        :return: ℝ+
        """
        return np.maximum(0, neurons)

    def get_gradient(self, gradient):
        """
        a = ReLU(z)
        ∂a/∂z = ReLU'(z), where ReLU'(z) -> 1 | z > 1, 0 | z <= 0

        ∂C/∂z = ∂C/∂a * ∂a/∂z = gradient * ReLU'(z)  (in the sense of backpropagation)

        :param gradient: ∂C/∂a, "cumulative" gradient of the next layer
        :return: ∂C/∂z
        """
        relu_d = self.input.copy()
        relu_d[relu_d <= 0] = 0
        relu_d[relu_d > 0] = 1
        gradient = gradient * relu_d
        return {}, gradient


class LeakyReLU(Activation):

    def __init__(self, leak=0.01):
        self.leak = leak
        super().__init__()

    def __repr__(self):
        if not self.built:
            return f"<{self.__class__.__qualname__}({self.leak}) {hex(id(self))} (?->{self.output_shape or '?'}) NOT BUILT>"
        return f"<{self.__class__.__qualname__}({self.leak}) {hex(id(self))} " \
               f"({self.input_shape}->{self.output_shape}) {self.n_param} Params>"

    def get_output(self, neurons):
        """
            ⎧ x, x > 0
        x = ⎨
            ⎩ leak * x, x < 0

        :param neurons: input
        :return: ℝ
        """
        return np.maximum(self.leak * neurons, neurons)

    def get_gradient(self, gradient):
        """
        a = lr(z)
        ∂a/∂z = lr'(z), where lr'(z) -> 1 | z > 1, leak | z <= 0

        ∂C/∂z = ∂C/∂a * ∂a/∂z = gradient * lr'(z)  (in the sense of backpropagation)

        :param gradient: ∂C/∂a, "cumulative" gradient of the next layer
        :return: ∂C/∂z
        """
        d_leaky = np.where(self.input > 0, 1, self.leak)
        gradient = d_leaky * gradient
        return {}, gradient
