import numpy as np

from .base_optimizer import Optimizer


class SGD(Optimizer):
    def step(self):
        for layer, trainable_gradient in zip(self.layers, self.trainable_gradients):
            for attr_name, grad in trainable_gradient.items():
                grad = grad * self.learning_rate / self.batch_size
                attr: np.ndarray = getattr(layer, attr_name)
                setattr(layer, attr_name, attr - grad)
