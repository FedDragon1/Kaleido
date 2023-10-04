import math
import unittest

from networks import *


class TestUtil(unittest.TestCase):

    def test_batcher(self):
        data = np.random.random([50, 3, 2])
        label = np.random.random(50)
        batched, length = batches(data, label, 12)
        batched = list(batched)

        print(batched)

        self.assertEqual(len(batched), math.ceil(len(data) / 12))

        first_data, first_label = batched[0]
        self.assertEqual(first_data.shape, (12, 3, 2))
        self.assertEqual(first_label.shape, (12,))

        last_data, last_label = batched[-1]
        self.assertEqual(last_data.shape, (2, 3, 2))
        self.assertEqual(last_label.shape, (2,))


class TestAssertions(unittest.TestCase):

    def test_valid_stride(self):
        from networks.util import assert_valid_stride

        assert_valid_stride(1, 3, "should not raise")
        assert_valid_stride((3, 4), 2, "should not raise")
        assert_valid_stride(3, 1, "should not raise")

        with self.assertRaises(TypeError):
            assert_valid_stride((3, 4), 3, "raises")

        with self.assertRaises(TypeError):
            assert_valid_stride((-1, 1, 3), 3, "raises")

        with self.assertRaises(TypeError):
            assert_valid_stride((3.5, 1), 2, "raises")

        with self.assertRaises(TypeError):
            assert_valid_stride(0, 1, "raises")
