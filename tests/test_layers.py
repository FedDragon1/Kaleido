import unittest

import numpy as np
from numpy.testing import assert_array_equal

from networks import *


class TestDense(unittest.TestCase):
    def test_forward(self):
        dense = Dense(8)
        input = np.random.random(10)
        output = dense(input)

        self.assertEqual(output.shape, (8,))
        self.assertEqual(dense.weights.shape, (8, 10))
        self.assertEqual(dense.biases.shape, (8,))

        expected = np.dot(dense.weights, input) + dense.biases
        assert_array_equal(expected, output)

    def test_backward(self):
        dense = Dense(8)
        input = np.random.random(10)
        output = dense(input)

        example_grad = np.ones(8)
        _, grad = dense.backprop(example_grad)

        assert_array_equal(grad, dense.weights.T @ example_grad)


class TestReshape(unittest.TestCase):
    def test_forward(self):
        reshape = Reshape((2, 3))

        original = np.arange(6)
        reshaped = reshape(original)
        expected = np.array([[0, 1, 2], [3, 4, 5]])

        assert_array_equal(reshaped, expected)

    def test_backward(self):
        reshape = Reshape((2, 3))

        original = np.array([0, 1, 2, 3, 4, 5])
        reshape(original)
        grad = np.array([[0, 1, 2], [3, 4, 5]])
        _, grad = reshape.backprop(grad)

        assert_array_equal(grad, original)
