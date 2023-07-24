import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal

from networks import *


class TestMSE(unittest.TestCase):

    def test_get_loss(self):
        mse = MSE()
        true = np.array([1, 0, 0, 0, 0])
        output = np.array([0.8, 0.1, 0.9, 0.3, 0.4])
        expected = 0.22200
        loss = mse.get_loss(output, true)
        self.assertAlmostEqual(expected, loss)

    def test_get_gradient(self):
        mse = MSE()
        true = np.array([1, 0, 0, 0, 0])
        output = np.array([0.8, 0.1, 0.9, 0.3, 0.4])
        expected = 2 * (output - true) / len(output)
        mse.forward(output, true)
        grad = mse.backprop()
        assert_array_equal(expected, grad)


class TestCrossEntropy(unittest.TestCase):

    def test_loss(self):
        ce = CrossEntropy()
        true = np.array([1, 0, 0, 0, 0])
        output = np.array([0.8, 0.1, 0.9, 0.3, 0.4])
        expected = -np.sum(true * np.log(output))
        loss = ce.forward(output, true)
        self.assertEqual(loss, expected)
