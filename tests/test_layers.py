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

    def test_forward_2(self):
        dense = Dense(10)
        test_x = np.arange(20)

        dense(test_x)

        # dense.weights = np.ones((10, 20))
        dense.weights = np.arange(10 * 20).reshape(10, 20)

        output = dense(test_x)

        # dummy_true = np.ones(10) * 10
        # loss = CrossEntropy()
        # grad = loss.get_gradient(output, dummy_true)

        # print(loss.get_loss(output, dummy_true))

        print(dense.backprop(np.ones_like(output)))

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


class TestConv1D(unittest.TestCase):
    def test_forward(self):
        conv = Conv1D(1, 3, padding="valid")
        test_x = np.array(
            [[0, 0], [3, 6], [4, 7], [5, 8], [1, 2], [4, 5], [2, 0], [0, 1]]
        )
        conv(test_x)
        conv.weights = np.array([[[1], [0]], [[0], [1]], [[2], [0]]])
        output = conv(test_x)
        expected = np.array([14, 20, 14, 15, 10, 4]).reshape(6, 1)

        assert_array_equal(output, expected)

    def test_forward_2(self):
        conv = Conv1D(2, 2)
        test_x = np.array([0, 3, 4, 5]).reshape(4, 1)
        conv(test_x)
        # (num_filter, k_size, num_channels)
        # (2, 2, 1)
        # (k_size, num_channels, num_filter)
        # (2, 1, 2)
        conv.weights = np.array([[[1, 2]], [[3, 4]]])
        test_y = np.array([[9, 12], [15, 22], [19, 28]])
        output = conv(test_x)

        assert_array_equal(output, test_y)

    def test_forward_3(self):
        conv = Conv1D(4, 3, padding="valid")
        test_x = np.array(
            [[0, 0], [3, 6], [4, 7], [5, 8], [1, 2], [4, 5], [2, 0], [0, 1]]
        )
        conv(test_x)
        conv.weights = np.array(
            [
                [[1, 0, 1, 2], [3, 2, -3, 0]],
                [[-2, 2, 3, 1], [1, 0, 1, 2]],
                [[3, 2, -3, 0], [-2, 2, 3, 1]],
            ]
        )
        output = conv(test_x)

        expected = np.array(
            [
                [-2, 28, 24, 22],
                [19, 46, 13, 32],
                [22, 30, 9, 31],
                [31, 36, -11, 20],
                [10, 16, 6, 16],
                [13, 16, -2, 11],
            ]
        )

        assert_array_equal(output, expected)

    def test_forward_4(self):
        conv = Conv1D(4, 3, padding="same")
        test_x = np.array(
            [[0, 0], [3, 6], [4, 7], [5, 8], [1, 2], [4, 5], [2, 0], [0, 1]]
        )
        conv(test_x)
        conv.weights = np.array(
            [
                [[1, 0, 1, 2], [3, 2, -3, 0]],
                [[-2, 2, 3, 1], [1, 0, 1, 2]],
                [[3, 2, -3, 0], [-2, 2, 3, 1]],
            ]
        )
        output = conv(test_x)

        expected = np.array(
            [
                [-3, 18, 9, 6],
                [-2, 28, 24, 22],
                [19, 46, 13, 32],
                [22, 30, 9, 31],
                [31, 36, -11, 20],
                [10, 16, 6, 16],
                [13, 16, -2, 11],
                [3, 0, 3, 6],
            ]
        )

        assert_array_equal(output, expected)

    def test_forward_5(self):
        conv = Conv1D(4, 3, padding="full")
        test_x = np.array(
            [[0, 0], [3, 6], [4, 7], [5, 8], [1, 2], [4, 5], [2, 0], [0, 1]]
        )
        conv(test_x)
        conv.weights = np.array(
            [
                [[1, 0, 1, 2], [3, 2, -3, 0]],
                [[-2, 2, 3, 1], [1, 0, 1, 2]],
                [[3, 2, -3, 0], [-2, 2, 3, 1]],
            ]
        )
        output = conv(test_x)

        expected = np.array(
            [
                [0, 0, 0, 0],
                [-3, 18, 9, 6],
                [-2, 28, 24, 22],
                [19, 46, 13, 32],
                [22, 30, 9, 31],
                [31, 36, -11, 20],
                [10, 16, 6, 16],
                [13, 16, -2, 11],
                [3, 0, 3, 6],
                [3, 2, -3, 0],
            ]
        )

        assert_array_equal(output, expected)

    def test_backpropagation(self):
        ...


class TestValid1D(unittest.TestCase):
    def test_build_raise(self):
        from networks.layers.conv.paddings import Valid1D

        valid = Valid1D(1, 2)

        with self.assertRaises(TypeError):
            list(valid.slices()())

    def test_slice1d(self):
        from networks.layers.conv.paddings import Valid1D

        valid = Valid1D(3, 1)
        valid.build(np.zeros((6, 2)))

        in_array = np.zeros((6, 2))
        processed = valid.pad(np.zeros((6, 2)))

        assert_array_equal(in_array, processed)

        for i, _slice in enumerate(valid.slices()()):
            self.assertEqual(_slice, slice(i, i + 3, None))

    def test_slice1d_stride(self):
        from networks.layers.conv.paddings import Valid1D

        valid = Valid1D(3, 2)
        valid.build(np.zeros((6, 2)))

        in_array = np.zeros((6, 2))
        processed = valid.pad(in_array)

        for i, _slice in enumerate(valid.slices()()):
            assert_array_equal(processed[_slice], processed[i * 2 : i * 2 + 3])

    def test_slice1d_even(self):
        from networks.layers.conv.paddings import Valid1D

        valid = Valid1D(2, 2)
        valid.build(np.zeros((6, 2)))

        for i, _slice in enumerate(valid.slices()()):
            self.assertEqual(_slice, slice(i * 2, i * 2 + 2))


class TestValid2D(unittest.TestCase):
    def test_slices2d(self):
        from networks.layers.conv.paddings import Valid2D

        input_array = np.array(
            [
                [[r("r00")], [r("r10")], [r("r20")]],
                [[r("r01")], [r("r11")], [r("r21")]],
                [[r("r02")], [r("r12")], [r("r22")]],
            ]
        )

        valid = Valid2D((2, 2), (1, 1))
        valid.build(input_array)

        expected = np.array(
            [
                [
                    [[[r("r00")], [r("r10")]], [[r("r01")], [r("r11")]]],
                    [[[r("r10")], [r("r20")]], [[r("r11")], [r("r21")]]],
                ],
                [
                    [[[r("r01")], [r("r11")]], [[r("r02")], [r("r12")]]],
                    [[[r("r11")], [r("r21")]], [[r("r12")], [r("r22")]]],
                ],
            ]
        )

        x_slices, y_slices = valid.slices()

        result = np.array(
            [
                [input_array[y_slice, x_slice] for x_slice in x_slices()]
                for y_slice in y_slices()
            ]
        )

        assert_array_equal(result, expected)

    def test_slice_2d_rectangular_stride(self):
        from networks.layers.conv.paddings import Valid2D

        input_array = np.array(
            [
                [[r("r00")], [r("r10")], [r("r20")], [r("r30")]],
                [[r("r01")], [r("r11")], [r("r21")], [r("r31")]],
                [[r("r02")], [r("r12")], [r("r22")], [r("r32")]],
                [[r("r03")], [r("r13")], [r("r23")], [r("r33")]],
            ]
        )

        valid = Valid2D((2, 2), (2, 1))
        valid.build(input_array)

        expected = np.array(
            [
                [  # y-axes: len 3
                    [  # x-axes: len 2
                        [[r("r00")], [r("r10")]],  # filter (2x2)
                        [[r("r01")], [r("r11")]],
                    ],
                    [[[r("r20")], [r("r30")]], [[r("r21")], [r("r31")]]],
                ],
                [
                    [[[r("r01")], [r("r11")]], [[r("r02")], [r("r12")]]],
                    [[[r("r21")], [r("r31")]], [[r("r22")], [r("r32")]]],
                ],
                [
                    [[[r("r02")], [r("r12")]], [[r("r03")], [r("r13")]]],
                    [[[r("r22")], [r("r32")]], [[r("r23")], [r("r33")]]],
                ],
            ]
        )

        x_slices, y_slices = valid.slices()

        result = np.array(
            [
                [input_array[y_slice, x_slice] for x_slice in x_slices()]
                for y_slice in y_slices()
            ]
        )

        assert_array_equal(expected, result)

    def test_slice_2d_rectangular_stride_rectangular_kernel(self):
        from networks.layers.conv.paddings import Valid2D

        input_array = np.array(
            [
                [[r("r00")], [r("r10")], [r("r20")], [r("r30")], [r("r40")]],
                [[r("r01")], [r("r11")], [r("r21")], [r("r31")], [r("r41")]],
                [[r("r02")], [r("r12")], [r("r22")], [r("r32")], [r("r42")]],
                [[r("r03")], [r("r13")], [r("r23")], [r("r33")], [r("r43")]],
            ]
        )

        valid = Valid2D((2, 3), (2, 1))
        valid.build(input_array)

        expected = np.array(
            [
                [
                    [
                        [[r("r00")], [r("r10")]],
                        [[r("r01")], [r("r11")]],
                        [[r("r02")], [r("r12")]],
                    ],
                    [
                        [[r("r20")], [r("r30")]],
                        [[r("r21")], [r("r31")]],
                        [[r("r22")], [r("r32")]],
                    ],
                ],
                [
                    [
                        [[r("r01")], [r("r11")]],
                        [[r("r02")], [r("r12")]],
                        [[r("r03")], [r("r13")]],
                    ],
                    [
                        [[r("r21")], [r("r31")]],
                        [[r("r22")], [r("r32")]],
                        [[r("r23")], [r("r33")]],
                    ],
                ],
            ]
        )

        x_slices, y_slices = valid.slices()

        result = np.array(
            [
                [input_array[y_slice, x_slice] for x_slice in x_slices()]
                for y_slice in y_slices()
            ]
        )

        assert_array_equal(expected, result)

    def test_slice_2d_rectangular_kernel(self):
        from networks.layers.conv.paddings import Valid2D

        input_array = np.array(
            [
                [[r("r00")], [r("r10")], [r("r20")], [r("r30")], [r("r40")]],
                [[r("r01")], [r("r11")], [r("r21")], [r("r31")], [r("r41")]],
                [[r("r02")], [r("r12")], [r("r22")], [r("r32")], [r("r42")]],
                [[r("r03")], [r("r13")], [r("r23")], [r("r33")], [r("r43")]],
                [[r("r04")], [r("r14")], [r("r24")], [r("r34")], [r("r44")]],
            ]
        )

        valid = Valid2D((2, 3), (2, 2))
        valid.build(input_array)

        expected = np.array(
            [
                [
                    [
                        [[r("r00")], [r("r10")]],
                        [[r("r01")], [r("r11")]],
                        [[r("r02")], [r("r12")]],
                    ],
                    [
                        [[r("r20")], [r("r30")]],
                        [[r("r21")], [r("r31")]],
                        [[r("r22")], [r("r32")]],
                    ],
                ],
                [
                    [
                        [[r("r02")], [r("r12")]],
                        [[r("r03")], [r("r13")]],
                        [[r("r04")], [r("r14")]],
                    ],
                    [
                        [[r("r22")], [r("r32")]],
                        [[r("r23")], [r("r33")]],
                        [[r("r24")], [r("r34")]],
                    ],
                ],
            ]
        )

        x_slices, y_slices = valid.slices()

        result = np.array(
            [
                [input_array[y_slice, x_slice] for x_slice in x_slices()]
                for y_slice in y_slices()
            ]
        )

        assert_array_equal(expected, result)


class TestValid3D(unittest.TestCase):
    def test_slide3d(self):
        from networks.layers.conv.paddings import Valid3D

        input_array = np.array(
            [
                [
                    [[r("r000")], [r("r100")], [r("r200")]],
                    [[r("r010")], [r("r110")], [r("r210")]],
                    [[r("r020")], [r("r120")], [r("r220")]],
                ],
                [
                    [[r("r001")], [r("r101")], [r("r201")]],
                    [[r("r011")], [r("r111")], [r("r211")]],
                    [[r("r021")], [r("r121")], [r("r221")]],
                ],
                [
                    [[r("r002")], [r("r102")], [r("r202")]],
                    [[r("r012")], [r("r112")], [r("r212")]],
                    [[r("r022")], [r("r122")], [r("r222")]],
                ],
            ]
        )

        valid = Valid3D((2, 2, 2), (1, 1, 1))
        valid.build(input_array)

        expected = np.array(
            [
                [  # z-axis: 2
                    [  # y-axis: 2
                        [  # x-axis: 2  | base(0, 0, 0)
                            # kernel: (2x2x2)
                            [  # kernel z-axis: 2
                                [[r("r000")], [r("r100")]],  # kernel y&x: (2x2)
                                [[r("r010")], [r("r110")]],
                            ],
                            [[[r("r001")], [r("r101")]], [[r("r011")], [r("r111")]]],
                        ],
                        [  # base(1, 0, 0)
                            [  # kernel z-axis: 2
                                [[r("r100")], [r("r200")]],  # kernel y&x: (2x2)
                                [[r("r110")], [r("r210")]],
                            ],
                            [[[r("r101")], [r("r201")]], [[r("r111")], [r("r211")]]],
                        ],
                    ],
                    [
                        [  # base(0, 1, 0)
                            [  # kernel z-axis: 2
                                [[r("r010")], [r("r110")]],  # kernel y&x: (2x2)
                                [[r("r020")], [r("r120")]],
                            ],
                            [[[r("r011")], [r("r111")]], [[r("r021")], [r("r121")]]],
                        ],
                        [  # base(1, 1, 0)
                            [  # kernel z-axis: 2
                                [[r("r110")], [r("r210")]],  # kernel y&x: (2x2)
                                [[r("r120")], [r("r220")]],
                            ],
                            [[[r("r111")], [r("r211")]], [[r("r121")], [r("r221")]]],
                        ],
                    ],
                ],
                [
                    [
                        [  # base(0, 0, 1)
                            [  # kernel z-axis: 2
                                [[r("r001")], [r("r101")]],  # kernel y&x: (2x2)
                                [[r("r011")], [r("r111")]],
                            ],
                            [[[r("r002")], [r("r102")]], [[r("r012")], [r("r112")]]],
                        ],
                        [  # base(1, 0, 1)
                            [  # kernel z-axis: 2
                                [[r("r101")], [r("r201")]],  # kernel y&x: (2x2)
                                [[r("r111")], [r("r211")]],
                            ],
                            [[[r("r102")], [r("r202")]], [[r("r112")], [r("r212")]]],
                        ],
                    ],
                    [
                        [  # base(0, 1, 1)
                            [  # kernel z-axis: 2
                                [[r("r011")], [r("r111")]],  # kernel y&x: (2x2)
                                [[r("r021")], [r("r121")]],
                            ],
                            [[[r("r012")], [r("r112")]], [[r("r022")], [r("r122")]]],
                        ],
                        [  # base(1, 1, 1)
                            [  # kernel z-axis: 2
                                [[r("r111")], [r("r211")]],  # kernel y&x: (2x2)
                                [[r("r121")], [r("r221")]],
                            ],
                            [[[r("r112")], [r("r212")]], [[r("r122")], [r("r222")]]],
                        ],
                    ],
                ],
            ]
        )

        x_slices, y_slices, z_slices = valid.slices()

        result = np.array(
            [
                [
                    [input_array[z_slice, y_slice, x_slice] for x_slice in x_slices()]
                    for y_slice in y_slices()
                ]
                for z_slice in z_slices()
            ]
        )

        assert_array_equal(expected, result)

    def test_slide3d_rectangular_kernel(self):
        from networks.layers.conv.paddings import Valid3D

        input_array = np.array(
            [
                [
                    [[r("r000")], [r("r100")], [r("r200")], [r("r300")]],
                    [[r("r010")], [r("r110")], [r("r210")], [r("r310")]],
                    [[r("r020")], [r("r120")], [r("r220")], [r("r320")]],
                    [[r("r030")], [r("r130")], [r("r230")], [r("r330")]],
                ],
                [
                    [[r("r001")], [r("r101")], [r("r201")], [r("r301")]],
                    [[r("r011")], [r("r111")], [r("r211")], [r("r311")]],
                    [[r("r021")], [r("r121")], [r("r221")], [r("r321")]],
                    [[r("r031")], [r("r131")], [r("r231")], [r("r331")]],
                ],
                [
                    [[r("r002")], [r("r102")], [r("r202")], [r("r302")]],
                    [[r("r012")], [r("r112")], [r("r212")], [r("r312")]],
                    [[r("r022")], [r("r122")], [r("r222")], [r("r322")]],
                    [[r("r032")], [r("r132")], [r("r232")], [r("r332")]],
                ],
                [
                    [[r("r003")], [r("r103")], [r("r203")], [r("r303")]],
                    [[r("r013")], [r("r113")], [r("r213")], [r("r313")]],
                    [[r("r023")], [r("r123")], [r("r223")], [r("r323")]],
                    [[r("r033")], [r("r133")], [r("r233")], [r("r333")]],
                ],
            ]
        )

        valid = Valid3D((2, 3, 2), (2, 1, 1))
        valid.build(input_array)

        expected = np.array(
            [
                [  # z-axis: 2
                    [  # y-axis: 2
                        [  # x-axis: 2  | base(0, 0, 0)
                            # kernel: (2x3x2)
                            [  # kernel z-axis: 2
                                [[r("r000")], [r("r100")]],  # kernel y&x: (3x2)
                                [[r("r010")], [r("r110")]],
                                [[r("r020")], [r("r120")]],
                            ],
                            [
                                [[r("r001")], [r("r101")]],
                                [[r("r011")], [r("r111")]],
                                [[r("r021")], [r("r121")]],
                            ],
                        ],
                        [  # base(2, 0, 0)
                            [  # kernel z-axis: 2
                                [[r("r200")], [r("r300")]],  # kernel y&x: (3x2)
                                [[r("r210")], [r("r310")]],
                                [[r("r220")], [r("r320")]],
                            ],
                            [
                                [[r("r201")], [r("r301")]],
                                [[r("r211")], [r("r311")]],
                                [[r("r221")], [r("r321")]],
                            ],
                        ],
                    ],
                    [
                        [  # base(0, 1, 0)
                            [  # kernel z-axis: 2
                                [[r("r010")], [r("r110")]],  # kernel y&x: (3x2)
                                [[r("r020")], [r("r120")]],
                                [[r("r030")], [r("r130")]],
                            ],
                            [
                                [[r("r011")], [r("r111")]],
                                [[r("r021")], [r("r121")]],
                                [[r("r031")], [r("r131")]],
                            ],
                        ],
                        [  # base(2, 1, 0)
                            [  # kernel z-axis: 2
                                [[r("r210")], [r("r310")]],  # kernel y&x: (3x2)
                                [[r("r220")], [r("r320")]],
                                [[r("r230")], [r("r330")]],
                            ],
                            [
                                [[r("r211")], [r("r311")]],
                                [[r("r221")], [r("r321")]],
                                [[r("r231")], [r("r331")]],
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [  # base(0, 0, 1)
                            [  # kernel z-axis: 2
                                [[r("r001")], [r("r101")]],  # kernel y&x: (3x2)
                                [[r("r011")], [r("r111")]],
                                [[r("r021")], [r("r121")]],
                            ],
                            [
                                [[r("r002")], [r("r102")]],
                                [[r("r012")], [r("r112")]],
                                [[r("r022")], [r("r122")]],
                            ],
                        ],
                        [  # base(2, 0, 1)
                            [  # kernel z-axis: 2
                                [[r("r201")], [r("r301")]],  # kernel y&x: (3x2)
                                [[r("r211")], [r("r311")]],
                                [[r("r221")], [r("r321")]],
                            ],
                            [
                                [[r("r202")], [r("r302")]],
                                [[r("r212")], [r("r312")]],
                                [[r("r222")], [r("r322")]],
                            ],
                        ],
                    ],
                    [
                        [  # base(0, 1, 1)
                            [  # kernel z-axis: 2
                                [[r("r011")], [r("r111")]],  # kernel y&x: (3x2)
                                [[r("r021")], [r("r121")]],
                                [[r("r031")], [r("r131")]],
                            ],
                            [
                                [[r("r012")], [r("r112")]],
                                [[r("r022")], [r("r122")]],
                                [[r("r032")], [r("r132")]],
                            ],
                        ],
                        [  # base(2, 1, 1)
                            [  # kernel z-axis: 2
                                [[r("r211")], [r("r311")]],  # kernel y&x: (3x2)
                                [[r("r221")], [r("r321")]],
                                [[r("r231")], [r("r331")]],
                            ],
                            [
                                [[r("r212")], [r("r312")]],
                                [[r("r222")], [r("r322")]],
                                [[r("r232")], [r("r332")]],
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [  # base(0, 0, 2)
                            [  # kernel z-axis: 2
                                [[r("r002")], [r("r102")]],  # kernel y&x: (3x2)
                                [[r("r012")], [r("r112")]],
                                [[r("r022")], [r("r122")]],
                            ],
                            [
                                [[r("r003")], [r("r103")]],
                                [[r("r013")], [r("r113")]],
                                [[r("r023")], [r("r123")]],
                            ],
                        ],
                        [  # base(2, 0, 2)
                            [  # kernel z-axis: 2
                                [[r("r202")], [r("r302")]],  # kernel y&x: (3x2)
                                [[r("r212")], [r("r312")]],
                                [[r("r222")], [r("r322")]],
                            ],
                            [
                                [[r("r203")], [r("r303")]],
                                [[r("r213")], [r("r313")]],
                                [[r("r223")], [r("r323")]],
                            ],
                        ],
                    ],
                    [
                        [  # base(0, 1, 2)
                            [  # kernel z-axis: 2
                                [[r("r012")], [r("r112")]],  # kernel y&x: (3x2)
                                [[r("r022")], [r("r122")]],
                                [[r("r032")], [r("r132")]],
                            ],
                            [
                                [[r("r013")], [r("r113")]],
                                [[r("r023")], [r("r123")]],
                                [[r("r033")], [r("r133")]],
                            ],
                        ],
                        [  # base(2, 1, 2)
                            [  # kernel z-axis: 2
                                [[r("r212")], [r("r312")]],  # kernel y&x: (3x2)
                                [[r("r222")], [r("r322")]],
                                [[r("r232")], [r("r332")]],
                            ],
                            [
                                [[r("r213")], [r("r313")]],
                                [[r("r223")], [r("r323")]],
                                [[r("r233")], [r("r333")]],
                            ],
                        ],
                    ],
                ],
            ]
        )

        x_slices, y_slices, z_slices = valid.slices()

        result = np.array(
            [
                [
                    [input_array[z_slice, y_slice, x_slice] for x_slice in x_slices()]
                    for y_slice in y_slices()
                ]
                for z_slice in z_slices()
            ]
        )

        assert_array_equal(expected, result)


class TestFull1D(unittest.TestCase):
    def test_full1d(self):
        from networks.layers.conv.paddings import Full1D

        kernel_size = 3
        stride = 1

        full = Full1D(kernel_size, stride)

        input_array = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[0, 0], [0, 0], [1, 2], [3, 4], [5, 6], [0, 0], [0, 0]])
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

        unpadded = full.unpad(processed)
        assert_array_equal(input_array, unpadded)

    def test_full1d_stride(self):
        from networks.layers.conv.paddings import Full1D

        kernel_size = 3
        stride = 2

        full = Full1D(kernel_size, stride)

        input_array = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[0, 0], [0, 0], [1, 2], [3, 4], [5, 6]])
        full.build(input_array)
        processed = full.pad(input_array)
        print(processed)
        assert_array_equal(expected, processed)

        slice_expected = np.array([[[0, 0], [0, 0], [1, 2]], [[1, 2], [3, 4], [5, 6]]])

        sliced = np.array(list(processed[x_slice] for x_slice in full.slices()()))
        assert_array_equal(sliced, slice_expected)

    def test_full1d_stride_2(self):
        from networks.layers.conv.paddings import Full1D

        kernel_size = 2
        stride = 3

        full = Full1D(kernel_size, stride)

        input_array = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[0, 0], [1, 2], [3, 4], [5, 6], [0, 0]])
        full.build(input_array)
        processed = full.pad(input_array)
        print(processed)
        assert_array_equal(expected, processed)


class TestFull2D(unittest.TestCase):
    def test_full2d(self):
        from networks.layers.conv.paddings import Full2D

        kernel_size = np.array([3, 3])
        stride = np.array([1, 2])

        full = Full2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
            ]
        )
        expected = np.array(
            [
                [[0], [0], [0], [0], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0], [0], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [0], [0], [0], [0], [0], [0]],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

        x_slices, y_slices = full.slices()

        for y_slice in y_slices():
            for x_slice in x_slices():
                print(processed[y_slice, x_slice, :])

        unpadded = full.unpad(expected)
        assert_array_equal(input_array, unpadded)

    def test_full2d_stride(self):
        from networks.layers.conv.paddings import Full2D

        kernel_size = np.array([3, 3])
        stride = np.array([1, 1])

        full = Full2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
            ]
        )
        expected = np.array(
            [
                [[0], [0], [0], [0], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0], [0], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [6], [6], [6], [6], [0], [0]],
                [[0], [0], [0], [0], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0], [0], [0], [0]],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

    def test_full2d_stride_2(self):
        from networks.layers.conv.paddings import Full2D

        kernel_size = np.array([3, 3])
        stride = np.array([2, 2])

        full = Full2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )
        expected = np.array(
            [
                [[0], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0]],
                [[0], [0], [6], [6], [6]],
                [[0], [0], [6], [6], [6]],
                [[0], [0], [6], [6], [6]],
                [[0], [0], [6], [6], [6]],
                [[0], [0], [0], [0], [0]],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

    def test_full2d_stride_3(self):
        from networks.layers.conv.paddings import Full2D

        kernel_size = np.array([2, 2])
        stride = np.array([3, 3])

        full = Full2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )
        expected = np.array(
            [
                [[0], [0], [0], [0], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

    def test_full2d_kernel_size(self):
        from networks.layers.conv.paddings import Full2D

        kernel_size = np.array([2, 1])
        stride = np.array([3, 3])

        full = Full2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )
        expected = np.array(
            [
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)


class TestFull3D(unittest.TestCase):
    def test_full3d(self):
        from networks.layers.conv.paddings import Full3D

        kernel_size = np.array([3, 3, 3])
        stride = np.array([1, 1, 1])

        full = Full3D(kernel_size, stride)

        input_array = np.array(
            [
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [6], [6], [6], [6], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0], [0], [0]],
                ],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)

    def test_full3d_stride(self):
        from networks.layers.conv.paddings import Full3D

        kernel_size = np.array([3, 3, 2])
        stride = np.array([2, 1, 1])

        full = Full3D(kernel_size, stride)

        input_array = np.array(
            [
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
            ]
        )
        expected = np.array(
            [
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [6], [6], [6]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
            ]
        )
        full.build(input_array)
        processed = full.pad(input_array)
        assert_array_equal(expected, processed)


class TestSame1D(unittest.TestCase):
    def test_same1d_no_stride(self):
        from networks.layers.conv.paddings import Same1D

        input_array = np.array([[1], [2], [3], [4]])
        kernel_size = 3
        stride = 1
        expected = np.array([[0], [1], [2], [3], [4], [0]])

        same = Same1D(kernel_size, stride)
        same.build(input_array)

        processed = same.pad(input_array)
        assert_array_equal(processed, expected)

    def test_same1d_no_stride_2(self):
        from networks.layers.conv.paddings import Same1D

        input_array = np.array([[1], [2], [3], [4], [5]])
        kernel_size = 4
        stride = 1
        expected = np.array([[0], [1], [2], [3], [4], [5], [0], [0]])

        same = Same1D(kernel_size, stride)
        same.build(input_array)

        processed = same.pad(input_array)
        print(processed)
        assert_array_equal(processed, expected)

    def test_same1d_stride(self):
        from networks.layers.conv.paddings import Same1D

        input_array = np.array([[1], [2], [3], [4], [5]])
        kernel_size = 3
        stride = 2
        expected = np.array([[0], [1], [2], [3], [4], [5], [0]])

        same = Same1D(kernel_size, stride)
        same.build(input_array)

        processed = same.pad(input_array)
        print(processed)
        assert_array_equal(processed, expected)

    def test_same1d_stride2(self):
        from networks.layers.conv.paddings import Same1D

        input_array = np.array([[1], [2], [3], [4], [5]])
        kernel_size = 3
        stride = 4
        expected = np.array([[0], [1], [2], [3], [4], [5], [0]])

        same = Same1D(kernel_size, stride)
        same.build(input_array)

        processed = same.pad(input_array)
        print(processed)
        assert_array_equal(processed, expected)

    def test_slicing(self):
        from networks.layers.conv.paddings import Same1D

        input_array = np.array([[1], [2], [3], [4], [5]])
        kernel_size = 3
        stride = 4
        expected = np.array([[[0], [1], [2]], [[4], [5], [0]]])

        same = Same1D(kernel_size, stride)
        same.build(input_array)
        to_be_sliced = same.pad(input_array)

        x_slices = same.slices()

        for expected, x_slice in zip(expected, x_slices()):
            assert_array_equal(expected, to_be_sliced[x_slice, :])


class TestSame2D(unittest.TestCase):
    def test_same2d(self):
        from networks.layers.conv.paddings import Same2D

        kernel_size = np.array([3, 3])
        stride = np.array([1, 2])

        same = Same2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
            ]
        )

        # output shape (y, x) = (2, 4)

        expected = np.array(
            [
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [0], [0], [0], [0], [0]],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)

        assert_array_equal(expected, processed)

        x_slices, y_slices = same.slices()

        for y_slice in y_slices():
            for x_slice in x_slices():
                print(processed[y_slice, x_slice, :])

    def test_same2d_stride(self):
        from networks.layers.conv.paddings import Same2D

        kernel_size = np.array([3, 3])
        stride = np.array([1, 1])

        same = Same2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
                [[6], [6], [6], [6]],
            ]
        )

        # output.shape (y, x) -> (4, 4)

        expected = np.array(
            [
                [[0], [0], [0], [0], [0], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [6], [0]],
                [[0], [0], [0], [0], [0], [0]],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)

    def test_same2d_stride_2(self):
        from networks.layers.conv.paddings import Same2D

        kernel_size = np.array([3, 3])
        stride = np.array([2, 2])

        same = Same2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )

        # output.shape (y, x) -> (2, 2)

        expected = np.array(
            [
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [6], [6], [6], [0]],
                [[0], [0], [0], [0], [0]],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)

    def test_same2d_stride_3(self):
        from networks.layers.conv.paddings import Same2D

        kernel_size = np.array([2, 2])
        stride = np.array([3, 3])

        same = Same2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )

        # output.shape (y, x) -> (2, 1)

        expected = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[0], [0], [0]],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)

    def test_same2d_kernel_size(self):
        from networks.layers.conv.paddings import Same2D

        kernel_size = np.array([2, 1])
        stride = np.array([3, 3])

        same = Same2D(kernel_size, stride)

        input_array = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )

        # output.shape (y, x) -> (2, 1)

        expected = np.array(
            [
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
                [[6], [6], [6]],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)


class TestSame3D(unittest.TestCase):
    def test_same3d(self):
        from networks.layers.conv.paddings import Same3D

        kernel_size = np.array([3, 3, 3])
        stride = np.array([1, 1, 1])

        same = Same3D(kernel_size, stride)

        input_array = np.array(
            [
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
                [
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                    [[6], [6], [6], [6]],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)

        unpadded = same.unpad(processed)
        assert_array_equal(unpadded, input_array)

    def test_same3d_stride(self):
        from networks.layers.conv.paddings import Same3D

        kernel_size = np.array([3, 3, 2])
        stride = np.array([2, 1, 1])

        same = Same3D(kernel_size, stride)

        input_array = np.array(
            [
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
                [[[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]], [[6], [6], [6]]],
            ]
        )

        # output.shape (z, y, x) -> (4, 3, 2)

        expected = np.array(
            [
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [6], [6], [6], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
                [
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                ],
            ]
        )
        same.build(input_array)
        processed = same.pad(input_array)
        assert_array_equal(expected, processed)


class TestMiscellaneous(unittest.TestCase):
    def test_tensor_dot(self):

        # k_size = 3, n_channels = 2, filter = 2 -> (3, 2, 2)

        fake_kernel = np.array(
            [
                [
                    [r("w111"), r("w112")],  # channel 1
                    [r("w121"), r("w122")],  # channel 2
                ],
                [
                    [r("w211"), r("w212")],  # channel 1
                    [r("w221"), r("w222")],  # channel 2
                ],
                [
                    [r("w311"), r("w312")],  # channel 1
                    [r("w321"), r("w322")],  # channel 2
                ],
            ]
        )

        # (k_size, output_length, n_channels) -> (3, 3, 2)
        dzdw_m = np.array(
            [
                [
                    [r("a11"), r("a12")],  # a1
                    [r("a21"), r("a22")],  # a2
                    [r("a31"), r("a32")],  # a3
                ],
                [
                    [r("a21"), r("a22")],  # a3
                    [r("a31"), r("a32")],  # a4
                    [r("a41"), r("a42")],  # a5
                ],
                [
                    [r("a31"), r("a32")],  # a3
                    [r("a41"), r("a42")],  # a4
                    [r("a51"), r("a52")],  # a5
                ],
            ]
        )
        # (output_length, n_channels) -> (3, 2)
        grad_m = np.array(
            [[r("z11"), r("z12")], [r("z21"), r("z22")], [r("z31"), r("z32")]]
        )

        # np.einsum("ij, j -> i", dzdw.T, grad) dot
        ret = np.einsum("ijk, ik -> jk", dzdw_m, grad_m)

        self.assertEqual(ret.shape, fake_kernel.shape[:-1])

    def test_tensor_dot_stride(self):

        # k_size = 3, n_channels = 2, filter = 3 -> (3, 2, 3)

        fake_kernel = np.array(
            [
                [
                    [r("w111"), r("w112"), r("w113")],  # channel 1
                    [r("w121"), r("w122"), r("w123")],  # channel 2
                ],
                [
                    [r("w211"), r("w212"), r("w213")],  # channel 1
                    [r("w221"), r("w222"), r("w223")],  # channel 2
                ],
                [
                    [r("w311"), r("w312"), r("w312")],  # channel 1
                    [r("w321"), r("w322"), r("w323")],  # channel 2
                ],
            ]
        )

        # (k_size, output_length, n_channels) -> (3, 2, 2)
        dzdw_m = np.array(
            [
                [
                    [r("a11"), r("a12")],  # a1
                    [r("a21"), r("a22")],  # a2
                    [r("a31"), r("a32")],  # a3
                ],
                # [     # strided
                #     [r("a21"), r("a22")],     # a3
                #     [r("a31"), r("a32")],     # a4
                #     [r("a41"), r("a42")]      # a5
                # ],
                [
                    [r("a31"), r("a32")],  # a3
                    [r("a41"), r("a42")],  # a4
                    [r("a51"), r("a52")],  # a5
                ],
            ]
        )
        # (output_length, n_channels) -> (2, 2)
        grad_m = np.array(
            [
                [r("z11"), r("z12")],
                [r("z21"), r("z22")],
            ]
        )

        # np.einsum("ij, j -> i", dzdw.T, grad) dot
        ret = np.einsum("ijk, ik -> jk", dzdw_m, grad_m)

        self.assertEqual(ret.shape, fake_kernel.shape[:-1])

    def test_bias_product(self):

        # output: (length, filters) -> (3, 2)
        grad_m = np.array(
            [
                [r("z11"), r("z12")],
                [r("z21"), r("z22")],
                [r("z31"), r("z32")],
            ]
        )

        # bias: (filters) -> 2
        biases = np.array([r("b1"), r("b2")])

        # dC/db = dC/db[:, j] for j in filters
        dcdb_expected = np.array([np.sum(grad_m[:, j]) for j in range(2)])

        vector_dcdb = np.sum(grad_m, axis=0)

        assert_array_equal(dcdb_expected, vector_dcdb)

    def test_dzda_slicing_ez(self):
        ez_w = np.array([r("w1"), r("w2"), r("w3")])
        ez_a = np.array([r("a1"), r("a2"), r("a3"), r("a4"), r("a5")])
        ez_z = np.array([r("z1"), r("z2"), r("z3")])

        # sample dzda should look like this
        # [[w1 w2 w3 0  0 ]
        #  [0  0  w1 w2 w3]]
        expected_dzda = np.array(
            [
                [r("w1"), r("w2"), r("w3"), r(0), r(0)],
                [r(0), r("w1"), r("w2"), r("w3"), r(0)],
                [r(0), r(0), r("w1"), r("w2"), r("w3")],
            ]
        )

        empty_dzda = np.zeros((3, 5))

        slices = [slice(0, 3), slice(1, 4), slice(2, 5)]

        for output_neuron_n, x_slice in enumerate(slices):
            empty_dzda[output_neuron_n, x_slice] = ez_w

        print(empty_dzda)

    def test_dzda_slicing(self):
        kernel_count = 2
        output_len = 3
        input_shape = (5, 4)
        kernel_size = 3

        w = np.arange(kernel_size * input_shape[1] * kernel_count).reshape(
            (kernel_size, input_shape[1], kernel_count)
        )

        z = np.ones((output_len, kernel_count))

        dzda = np.zeros((output_len, *input_shape, kernel_count))

        slices = [
            slice(0, 3),
            slice(1, 4),
            slice(2, 5)
        ]

        for output_i, x_slice in enumerate(slices):
            dzda[output_i, x_slice] = w

        dcda = np.einsum("ijkl, il -> jk", dzda, z)
        self.assertEqual(dcda.shape, input_shape)


class r:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s

    def __str__(self):
        return self.s

    def __add__(self, other):
        if other == 0:
            return self
        return r(f"{self.s} + {other.s if hasattr(other, 's') else other}")

    def __radd__(self, other):
        if other == 0:
            return self
        return r(f"{self.s} + {other.s if hasattr(other, 's') else other}")

    def __mul__(self, other):
        return r(f"{self.s}{other.s if hasattr(other, 's') else other}")

    def __eq__(self, other):
        return self.s == other.s if hasattr(other, "s") else other

    def __float__(self):
        return float(self.s[1:])
