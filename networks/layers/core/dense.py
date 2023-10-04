import numpy as np

from networks.layers.core.base_layers import Layer

from networks.util import assert_max_ndim, assert_valid_output_shape


class Dense(Layer):
    """Regular densely-connected layer, also known as Fully Connected layer"""

    def __init__(self, output_shape: int, **kwargs):
        output_shape = assert_valid_output_shape(output_shape)
        super().__init__(output_shape, **kwargs)

    def __repr__(self):
        if not self.built:
            return f"<{self.__class__.__qualname__}({self.output_shape}) " \
                   f"{hex(id(self))} (?->{self.output_shape or '?'}) NOT BUILT>"
        return f"<{self.__class__.__qualname__}({self.output_shape}) {hex(id(self))} " \
               f"({self.input_shape}->{self.output_shape}) {self.n_param} Params>"

    def get_output(self, neurons):
        """
        Forward pass
        z = w · a + b

        :param neurons: input neurons
        :return: fully connected output
        """
        output = self.weights @ neurons + self.biases
        return output

    def get_trainable(self):
        return {
            "weights": np.zeros((self.output_shape, self.input_shape)),
            "biases": np.zeros(self.output_shape),
        }

    def build_parameters(self, neurons):
        assert_max_ndim(neurons, 1, f"Expected vector input from {self}, got {neurons=}")

        self.input_shape = np.asarray(len(neurons))
        # n -> m, m rows n column weights
        self.weights = np.random.random([self.output_shape, self.input_shape]) - 0.5
        self.biases = np.zeros(self.output_shape)

        self.n_param = self.output_shape * self.input_shape + self.output_shape

        # built
        self.built = True

    def backprop(self, gradients):
        """
        Backpropagation for **this given data**
        Optimizers will handle the batch, this method should only be used internally

        target function:
            z = w · a + b

        weights:
            ∂z[j]/∂w[j][k] = ∂[w[j][1] * a[1] + w[j][2] * a[2] + ... + w[j][k] * a[k] + ... + b[j]] / ∂w[j][k]
                           = a[k]
            ∂C/∂w[j][k] = ∂C/∂z[j] * ∂z[j]/∂w[j][k]
                        = grad[j] * a[k]
            ∂C/∂w = [a[k] * grad[j] for k] for j      ** k->col & j->row **
                  = ⎡g[1]*a[1] g[1]*a[2] ... g[1]*a[k]⎤
                    ⎥g[2]*a[1] g[2]*a[2] ... g[1]*a[k]⎥
                    ⎥    ⋮         ⋮      ⋱      ⋮    ⎥
                    ⎣g[j]*a[1] g[j]*a[2] ... g[j]*a[k]⎦
                  = grad ⦻ a

        biases:
            ∂z[j]/∂b[j] = ∂[w[j][1] * a[1] + w[j][2] * a[2] + ... + b[j]] / b[j]
                        = 1
            ∂C/∂b[j] = ∂C/∂z[j] * ∂z[j]/∂b[j]
                     = grad[j]
            ∂C/∂b = grad

        inputs:
            ∂z/∂a = J(z)
                  = ⎡∂z[1]/∂a[1] ∂z[1]/∂a[2] ... ∂z[1]/∂a[k]⎤
                    ⎥∂z[2]/∂a[1] ∂z[2]/∂a[2] ... ∂z[2]/∂a[k]⎥
                    ⎥     ⋮           ⋮       ⋱       ⋮     ⎥
                    ⎣∂z[j]/∂a[1] ∂z[j]/∂a[2] ... ∂z[j]/∂a[k]⎦
                  = ⎡∂[w[1][1] * a[1]]/∂a[1]  ∂[w[1][2] * a[2]]/∂a[2]  ...  ∂[w[1][k] * a[k]]/∂a[k]⎤
                    ⎥∂[w[2][1] * a[1]]/∂a[1]  ∂[w[2][2] * a[2]]/∂a[2]  ...  ∂[w[2][k] * a[k]]/∂a[k]⎥
                    ⎥          ⋮                        ⋮               ⋱             ⋮            ⎥
                    ⎣∂[w[j][1] * a[1]]/∂a[1]  ∂[w[j][2] * a[2]]/∂a[2]  ...  ∂[w[j][k] * a[k]]/∂a[k]⎦
                  = w
            ∂C/∂a = [∂C/∂z]^T · ∂z/∂a
                  = grad^T · w

        :param gradients: gradients from next layer (usually local gradient from activation fn)
        :return: ({"weights": ∇weights, "biases": ∇biases},  ∇neurons)
        """

        # trainable parameters
        nabla_w = np.outer(gradients, self.input)
        nabla_b = gradients.copy()

        # gradients that get passed on backprop
        nabla_a = gradients.T @ self.weights

        trainable = {"weights": nabla_w, "biases": nabla_b}
        return trainable, nabla_a
