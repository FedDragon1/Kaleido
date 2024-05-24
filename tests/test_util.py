import math
import unittest

from networks import *
from networks.util import requires_build, requires_layer_build


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


class TestDecorators(unittest.TestCase):

    def test_assert_built(self):
        from networks.util import requires_build

        class T:
            def __init__(self, built):
                self.built = built

            @property
            @requires_layer_build
            def example_attr(self):
                return "123"

        t = T(True)

        self.assertEqual("123", t.example_attr)
        with self.assertRaises(TypeError):
            t = T(False)
            print(t.example_attr)

        try:
            t.example_attr
        except Exception as e:
            print(e)

    def test2(self):
        padding_requires_build = requires_build("Padding not built. Please build this padding first")

        @padding_requires_build
        def unpad1d(self, a):
            """
            Unpads the padded 2d array.

            :param padded_data: array from `pad`
            :return: unpadded 2d array
            """
            return f"p: {a}"

        class T:
            unpad = unpad1d

        t = T()
        t.built = False

        with self.assertRaises(TypeError):
            t.unpad(2)

        t.built = True
        self.assertEqual("p: 3", t.unpad(3))
