import numpy as np

from .base_activation import Activation


__all__ = ("SoftMax",)


class SoftMax(Activation):
    """
    Softmax is a mathematical function that converts a vector
    of numbers into a vector of probabilities, where the
    probabilities of each value are proportional to the
    relative scale of each value in the vector.

    [↑] https://machinelearningmastery.com/softmax-activation-function-with-python/
    """

    def get_output(self, neurons):
        """
        σ(z) = exp(z) / sum(exp(z))

        One property of softmax function is σ(z) = σ(z + n),
        to avoid overflow in `np.exp`, subtract the max value in
        the input neurons.

        :param neurons: input
        :return: softmax(neurons)
        """
        neurons -= np.max(neurons)
        exp_arr = np.exp(neurons)
        return exp_arr / np.sum(exp_arr)

    def get_gradient(self, gradient):
        """
        a = σ(z)
        ∂a/∂z = J(σ)
              = ⎡s[1]*(1 - s[1])    -s[1]*s[2]   ...    -s[1]*s[n]  ⎤   (where s is the output matrix)
                ⎥   -s[2]*s[1]   s[2]*(1 - s[2]) ...    -s[2]*s[n]  ⎥
                ⎥        ⋮               ⋮        ⋱          ⋮      ⎥
                ⎣   -s[n]*s[1]      -s[n]*s[2]   ... s[n]*(1 - s[n])⎦
              = ⎡s[1]    0   ...   0 ⎤   ⎡s[1]*s[1] s[1]*s[2] ... s[1]*s[n]⎤
                ⎥ 0     s[2] ...   0 ⎥ - ⎥s[2]*s[1] s[2]*s[2] ... s[2]*s[n]⎥
                ⎥ ⋮      ⋮    ⋱    ⋮ ⎥   ⎥    ⋮         ⋮      ⋱      ⋮    ⎥
                ⎣ 0      0   ... s[n]⎦   ⎣s[n]*s[1] s[n]*s[2] ... s[n]*s[n]⎦
              = diag(s) - s ⦻ s
        (proof see https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)

        ∂C/∂z = [∂C/∂a]^T · ∂a/∂z
              = [∂a/∂z]^T · ∂C/∂a
              = J(σ)^T · gradient
              = J(σ) · gradient  (Jacobian is orthogonal)
              (in the sense of backpropagation)

        :param gradient: ∂C/∂a, "cumulative" gradient of the next layer
        :return: ∂C/∂z
        """
        jacobian = np.diag(self.output) - np.outer(self.output, self.output)
        return {}, jacobian @ gradient
