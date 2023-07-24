import numpy as np

from .base_optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10e-8):
        """
        Require: learning_rate: stepsize
        Require: β1, β1 ∈ [0,1): exponential decay rates for the moment estimates
        Require: m0 <- 0 (Initialize 1st moment vector)  (passed in through `build`)
                 v0 <- 0 (Initialize 2nd moment vector)  (passed in through `build`)
                 t <- 0 (Initialize timestep)
        Require: f(θ): Stochastic objective function with parameters θ (passed in through `build`)

        :param learning_rate: stepsize
        :param beta1: exponential decay rate 1
        :param beta2: exponential decay rate 2
        :param epsilon: small constant to avoid division by 0
        """
        self.t = 0

        # Exponential decay rates
        self.old_beta1 = self.beta1 = beta1
        self.old_beta2 = self.beta2 = beta2

        # when t too big use 0
        self.beta1_use0 = self.beta2_use0 = False

        self.epsilon = epsilon
        super().__init__(learning_rate)

    def build(self, layers, batch_size):
        super().build(layers, batch_size)
        # moment vector
        # these two should be matching shape of self.trainable_parameters
        # caution on changing the architecture
        self.m = [
            list(layer.get_trainable().values()) for layer in self.layers
        ]  # first moment vec
        self.v = [
            list(layer.get_trainable().values()) for layer in self.layers
        ]  # second moment vec

    def step(self):
        """
        **All operations on vectors are element-wise**

        t <- t + 1
        g <- ∇f                             (Get gradients w.r.t stochastic objective at timestep t)
        m <- β1 * m + (1 - β1) * g           (Update biased first moment estimate)
        v <- β2 * v + (1 - β2) * g ** 2      (Update biased second raw moment estimate)
        m_hat <- m / (1 - β1 ** t)           (Compute bias-corrected first moment estimate)
        v_hat <- v / (1 - β2 ** t)           (Compute bias-corrected second raw moment estimate)
        θ -= learning_rate * m_hat / (sqrt(v_hat) + ε)  (Update Parameters)
        """

        self.t += 1

        for layer, trainable_gradient, layer_m, layer_v in zip(
            self.layers, self.trainable_gradients, self.m, self.v
        ):
            for i, ((attr_name, g), m, v) in enumerate(
                zip(trainable_gradient.items(), layer_m, layer_v)
            ):
                # modification to m and v are not inplace -
                # which means they need to be set back to layer_m and layer_v
                # assigning to these lists is safe
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * g ** 2

                if self.beta1_use0:
                    beta1_t = 0
                else:
                    self.old_beta1 = beta1_t = self.old_beta1 * self.beta1   # self.beta1 ** self.t
                    if beta1_t < 10e-4:
                        self.beta1_use0 = True

                if self.beta2_use0:
                    beta2_t = 0
                else:
                    self.old_beta2 = beta2_t = self.old_beta2 * self.beta2   # self.beta2 ** self.t
                    if beta2_t < 10e-4:
                        self.beta2_use0 = True

                m_hat = m / (1 - beta1_t)
                v_hat = v / (1 - beta2_t)

                # assign the new m and v values back to layer_m and layer_v
                layer_m[i] = m
                layer_v[i] = v

                decrement = self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
                attr: np.ndarray = getattr(layer, attr_name)
                setattr(layer, attr_name, attr - decrement)
