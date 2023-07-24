import math
import unittest

import numpy as np

from networks import *

from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestActivationSubclassHook(unittest.TestCase):

    def test_hook(self):
        self.assertIsInstance(
            Activation("relu"),
            ReLU
        )
        self.assertIsInstance(
            Activation("sOfTmAX"),
            SoftMax
        )
        with self.assertRaises(TypeError):
            Activation(" sigmoid")
        with self.assertRaises(TypeError):
            Activation()

        leaky = Activation("leakyrelu")
        self.assertIsInstance(leaky, LeakyReLU)
        leaky = Activation('leakyrelu', 0.1)
        self.assertIsInstance(leaky, LeakyReLU)
        self.assertEqual(leaky.leak, 0.1)
        leaky = Activation('leakyrelu', leak=0.1)
        self.assertIsInstance(leaky, LeakyReLU)
        self.assertEqual(leaky.leak, 0.1)


class TestReLU(unittest.TestCase):

    def test_forward(self):
        relu = ReLU()
        self.assertEqual(relu.forward(-5), 0)

        input = np.array([
                    [-2, -999, 3],
                    [10, 0, 8],
                    [-9, -7, -0.0001]
                ])

        expected = np.array([
                [0, 0, 3],
                [10, 0, 8],
                [0, 0, 0]
            ])
        assert_array_equal(
            relu.forward(input),
            expected
        )
        assert_array_equal(
            relu.output,
            expected
        )
        assert_array_equal(
            relu.input,
            input
        )

    def test_backward(self):
        relu = ReLU()
        input = np.array([
            [-2, -999, 3],
            [10, 0, 8],
            [-9, -7, -0.0001]
        ])
        gradient = np.ones((3, 3)) * 0.1
        expected = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0]
        ]) * gradient

        relu(input)
        trainable, neurons = (relu.get_gradient(
            gradient
        ))

        assert_array_equal(expected, neurons)
        self.assertEqual(trainable, {})


class TestSigmoid(unittest.TestCase):

    def test_forward(self):
        sigmoid = Sigmoid()
        input = np.array([
            [-1, 0, 1],
            [-100, 0, 100],
            [-2, -1, -0.5],
            [2, 1, 0.5]
        ])
        expected = np.array([
            [0.2689, 0.5, 0.7311],
            [0, 0.5, 1],
            [0.1192, 0.2689, 0.3775],
            [0.8808, 0.7311, 0.6225]
        ])
        pred = sigmoid(input)
        assert_array_almost_equal(pred, expected, 4)

    def test_backward(self):
        sigmoid = Sigmoid()
        input = np.array([
            [-1, 0, 1],
            [-100, 0, 100],
            [-2, -1, -0.5],
            [2, 1, 0.5]
        ])
        output = sigmoid(input)
        sigmoid_d = output * (1 - output)
        gradient = np.ones((4, 3)) * 2
        expected = sigmoid_d * 2
        _, pred = sigmoid.get_gradient(gradient)
        assert_array_equal(expected, pred)


class TestTanh(unittest.TestCase):

    def test_forward(self):
        input = np.array([
            [-1, 0, 1],
            [-100, 0, 1],
            [-2, 2, 3]
        ])
        expected = np.array([
            [math.tanh(-1), math.tanh(0), math.tanh(1)],
            [math.tanh(-100), math.tanh(0), math.tanh(1)],
            [math.tanh(-2), math.tanh(2), math.tanh(3)]
        ])
        pred = Tanh()(input)
        assert_array_almost_equal(expected, pred)

    def test_backward(self):
        input = np.array([
            [-1, 0, 1],
            [-100, 0, 1],
            [-2, 2, 3]
        ])
        tanh_d = -np.tanh(input) ** 2 + 1
        grad = np.ones((3, 3)) * 0.5

        expected = tanh_d * grad
        tanh = Tanh()
        tanh(input)
        _, pred = tanh.get_gradient(grad)
        assert_array_equal(expected, pred)


class TestLeakyReLU(unittest.TestCase):

    def test_forward(self):
        input = np.array([
            [-5, -1, 0],
            [2, 6, -10],
            [-4.5, 8, 0]
        ])
        expected = np.array([
            [-0.5, -0.1, 0],
            [2, 6, -1],
            [-0.45, 8, 0]
        ])
        leaky = Activation("leakyrelu", 0.1)
        pred = leaky(input)

        assert_array_equal(expected, pred)

        expected[expected < 0] /= 10
        leaky = Activation("leakyrelu")
        pred = leaky(input)
        assert_array_equal(expected, pred)

    def test_backward(self):
        input = np.array([
            [-5, -1, 0],
            [2, 6, -10],
            [-4.5, 8, 0]
        ])
        d_leaky = np.array([
            [0.01, 0.01, 0.01],
            [1, 1, 0.01],
            [0.01, 1, 0.01]
        ])
        gradient = np.ones((3, 3)) * 2
        expected = gradient * d_leaky

        leaky = LeakyReLU()
        leaky(input)
        _, pred = leaky.get_gradient(gradient)

        assert_array_equal(d_leaky * gradient, pred)


class TestSoftMax(unittest.TestCase):

    def test_forward(self):
        input = np.array([3, 4, 1])
        expected = np.array([0.25949646034242, 0.70538451269824, 0.03511902695934])
        pred = SoftMax()(input)

        assert_array_almost_equal(expected, pred)
