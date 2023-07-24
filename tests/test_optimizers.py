import unittest

import numpy as np

from networks import *


class TestSGD(unittest.TestCase):

    def test_initialize(self):
        np.random.seed(1)
        model = Sequential(
            Input(10),
            Dense(8),
            Activation('leakyrelu'),
            Dense(6),
            Activation('softmax')
        )
        optimizer = SGD()
        loss = MSE()

        model.compile(optimizer=optimizer, loss=loss)

        model.summary()
        expected = np.array([1, 0, 0, 0, 0, 0])
        output = model(np.ones(10))

        loss.forward(output, expected)
        grad = loss.backprop()
        optimizer.collect(grad)
        # print(optimizer.trainable_gradients)

        optimizer.step()
