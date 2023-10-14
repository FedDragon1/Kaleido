import math
from abc import abstractmethod

import numpy as np

from ...util import SubclassDispatcherMeta, assert_built


def unpad1d(self, padded_data):
    """
    Unpads the padded 2d array.

    :param padded_data: array from `pad`
    :return: unpadded 2d array
    """
    assert_built(self, "Padding not built. Please build this padding first")

    (x_slice,) = self.position
    return padded_data[x_slice]


def unpad2d(self, padded_data):
    """
    Unpads the padded 3d array.

    :param padded_data: array from `pad`
    :return: unpadded 3d array
    """
    assert_built(self, "Padding not built. Please build this padding first")

    x_slice, y_slice = self.position
    return padded_data[y_slice, x_slice]


def unpad3d(self, padded_data):
    """
    Unpads the padded 4d array.

    :param padded_data: array from `pad`
    :return: unpadded 4d array
    """
    assert_built(self, "Padding not built. Please build this padding first")

    x_slice, y_slice, z_slice = self.position
    return padded_data[z_slice, y_slice, x_slice]


def slices1d(self):
    """
    Function returning 1 generator, producing slices along x-axis.
    Built this way to match the signature with other slices.

    :return: x_slices_generator -> slice
    """
    assert_built(self, "Padding not built. Please build this padding first")

    def x_slices_generator():
        # end + 1 to make it inclusive
        for start in range(self.start, self.end + 1, self.stride):
            # slice produces range [start, end), so adding kernel_size is valid
            yield slice(start, start + self.kernel_size)

    return x_slices_generator


def slices2d(self):
    """
    Function returning 2 generators, producing slices
    on x and y axes respectively.

    :return: x_slices_generator -> slice, y_slices_generator -> slice
    """
    assert_built(self, "Padding not built. Please build this padding first")

    def x_slices_generator():
        for x_start in range(self.start[0], self.end[0] + 1, self.stride[0]):
            yield slice(x_start, x_start + self.kernel_size[0])

    def y_slices_generator():
        for y_start in range(self.start[1], self.end[1] + 1, self.stride[1]):
            yield slice(y_start, y_start + self.kernel_size[1])

    return x_slices_generator, y_slices_generator


def slices3d(self):
    """
    Same as slices2d but produces slices on
    x, y, and z axes respectively.

    :return: (
        x_slices_generator -> slice,
        y_slices_generator -> slice,
        z_slices_generator -> slice
    )
    """
    assert_built(self, "Padding not built. Please build this padding first")

    def x_slices_generator():
        for x_start in range(self.start[0], self.end[0] + 1, self.stride[0]):
            yield slice(x_start, x_start + self.kernel_size[0])

    def y_slices_generator():
        for y_start in range(self.start[1], self.end[1] + 1, self.stride[1]):
            yield slice(y_start, y_start + self.kernel_size[1])

    def z_slices_generator():
        for z_start in range(self.start[2], self.end[2] + 1, self.stride[2]):
            yield slice(z_start, z_start + self.kernel_size[2])

    return x_slices_generator, y_slices_generator, z_slices_generator


class Padding(metaclass=SubclassDispatcherMeta):
    """Base Padding Layer"""

    def __init__(self, kernel_size, stride):
        """
        Creates a padding. Parameter should be provided
        by convolution layers, so they are sanitized.

        After instantiating a padding object one can
        immediately start processing (padding) data, but
        building the layer is necessary for the slices
        for convolution kernels.

        All attributes of a padding object:
            - `kernel_size`     Sequence[n]
            - `stride`          Sequence[n]
            - `built`           bool
            - `start`           np.zeros[n]      REQUIRES BUILD
            - `end`             np.ndarray[n]    REQUIRES BUILD
            - `processed_shape` tuple[int]       REQUIRES BUILD
            - `position`        tuple[slice]     REQUIRES BUILD
        Note: `[n]` means length of `n`.

        The purpose of making initialization in 2 stages is to
        match behavior with other layers. At the time of
        `__init__` is called there is no information on the
        input shape of the layer. Thus, `build` method should
        be called within `Conv.build` method.

        Similarly, the purpose of getting kernel size and stride
        in `__init__` is for the convenience of `build`.
        Specifically, this allows `process` to be called before
        `build` is performed. More detail is provided in `build`.

        :param kernel_size: kernel size array
        :param stride: stride array
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.built = False

    def __repr__(self):
        if not self.built:
            return f"<{self.__class__.__qualname__}() NOT BUILT>"
        return f"<{self.__class__.__qualname__}()>"

    def build(self, raw_data):
        """
        Builds the padding layer.

        This method should be called as `Conv.build` is called,
        passing the *raw*, *unmodified* neuron data. The soul
        purpose of `build` is to determine the `end` attribute,
        which is used to produce slices of data that matches the
        size of kernel.

        This method is shared along all kind of paddings since
        `end` depends on the *processed* neurons. This is the
        reason that `process` should be independent of `build`.

        This method may be overriden to add padding specific
        attributes. Always call this method in the end of
        customization.

        :param raw_data: neurons
        :return: None
        """

        data = self.pad(raw_data)
        dimension = data.ndim - 1

        # first starting index of convolution, always 0 vector
        if dimension == 1:
            self.start = 0
        else:
            self.start = np.zeros(dimension, dtype=np.int32)

        # last starting index of convolution
        # solve this system of equations
        # end % stride == 0     # must be a step on strides
        # end + k_size <= len   # right side constrain

        if dimension == 1:
            end = len(data) - self.kernel_size
        else:
            # inverse the shape, so "x-axis" is the first element
            # i.e (z, y, x) -> (x, y, z)
            end = (
                np.array(data.shape[:-1][::-1]) - self.kernel_size
            )  # ensure the convolution can be performed at `end`
        self.end = end - end % self.stride  # must be a step on strides
        self.built = True

    @abstractmethod
    def unpad(self, padded_data):
        """Undo the pad, returning original data"""
        ...

    @abstractmethod
    def pad(self, raw_data):
        """Process the data and return padded data"""
        ...

    @abstractmethod
    def slices(self):
        ...


class Full(Padding):
    """
    The Full padding, where 0 is padded around the
    input so that first convolution only contains
    """


class Full1D(Full):
    """
    Full1D, pads 0 around the start and end of the sequence
    for all the channels.
    """

    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        :param raw_data: unprocessed data
        :return: None
        """
        length, num_channel = raw_data.shape
        left_padding = self.kernel_size - 1

        left_padding_len = length + left_padding

        # test if we don't need right padding (because the length is perfect)
        if (
            left_padding_len // self.stride - 1
        ) * self.stride + self.kernel_size == left_padding_len:
            desired_len = left_padding_len
        else:
            desired_chunks = math.ceil(left_padding_len / self.stride)
            desired_len = (desired_chunks - 1) * self.stride + self.kernel_size

        self.processed_shape = (desired_len, num_channel)
        self.position = (slice(left_padding, left_padding + length),)

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Examples:
            array:      [A B C D E F G]
            kernel:     [1 2 3]
            stride:     1
            processed:  [0 0 A B C D E F G 0 0]

            array:      [A B C D E F G H]
            kernel:     [1 2 3]
            stride:     2
            processed:  [0 0 A B C D E F G H 0]

        :param raw_data: unprocessed 2D tensor (time, #channel)
        :return: 0 padded array
        """
        processed = np.zeros(self.processed_shape)
        # all channels
        (x_slice,) = self.position
        processed[x_slice, :] = raw_data
        return processed

    slices = slices1d

    unpad = unpad1d


class Full2D(Full):
    """
    Full2D, pads 0 around the input feature map for all channels.
    Expects a 3D array input (y, x, channel)
    """

    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        Notice that `position` will be a tuple since there are
        2 slices in y-axis and x-axis respectively

        :param raw_data: unprocessed data (3D tensor)
        :return: None
        """
        *size, num_channel = raw_data.shape
        size = np.array(size)

        # requires kernel_size to be ndarray
        # size -> (y, x, channel) where as k_size -> (x, y)
        # top_left should match with size (y, x)
        top_left_padding = self.kernel_size[::-1] - 1

        y_length, x_length = size + top_left_padding

        # test if the x-length is not perfect, pad accordingly
        if not (
            (x_length // self.stride[0] - 1) * self.stride[0] + self.kernel_size[0]
            == x_length
        ):
            desired_chunks_x = math.ceil(x_length / self.stride[0])
            x_length = (desired_chunks_x - 1) * self.stride[0] + self.kernel_size[0]

        if not (
            (y_length // self.stride[1] - 1) * self.stride[1] + self.kernel_size[1]
            == y_length
        ):
            desired_chunks = math.ceil(y_length / self.stride[1])
            y_length = (desired_chunks - 1) * self.stride[1] + self.kernel_size[1]

        self.processed_shape = (y_length, x_length, num_channel)
        self.position = (
            # x-axis, y-axis
            slice(top_left_padding[1], top_left_padding[1] + size[1]),
            slice(top_left_padding[0], top_left_padding[0] + size[0]),
        )

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Examples:
            array:      [[A B C D]
                         [E F G H]
                         [I J K L]
                         [M N O P]]
            kernel:     [[1 2 3]
                         [4 5 6]
                         [7 8 9]]
            stride:     (1, 1)
            processed:  [[0 0 0 0 0 0 0 0]
                         [0 0 0 0 0 0 0 0]
                         [0 0 A B C D 0 0]
                         [0 0 E F G H 0 0]
                         [0 0 I J K L 0 0]
                         [0 0 M N O P 0 0]
                         [0 0 0 0 0 0 0 0]
                         [0 0 0 0 0 0 0 0]]

            array:      [[A B C D]
                         [E F G H]
                         [I J K L]
                         [M N O P]]
            kernel:     [[[1 2 3]
                         [4 5 6]
                         [7 8 9]]
            stride:     (2, 3)
            processed:  [[0 0 0 0 0 0 0]
                         [0 0 0 0 0 0 0]
                         [0 0 A B C D 0]
                         [0 0 E F G H 0]
                         [0 0 I J K L 0]
                         [0 0 M N O P 0]]

        :param raw_data: unprocessed 3D tensor (#y, #x, #channel)
        :return: 0 padded array
        """

        processed = np.zeros(self.processed_shape)
        x_slice, y_slice = self.position
        # all channels
        processed[y_slice, x_slice, :] = raw_data
        return processed

    slices = slices2d

    unpad = unpad2d


class Full3D(Full):
    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        Notice that `position` will be a tuple since there are
        3 slices in z-axis, y-axis and x-axis respectively

        :param raw_data: unprocessed data (4D tensor)
        :return: None
        """
        *size, num_channel = raw_data.shape
        size = np.array(size)

        # requires kernel_size to be ndarray
        base_padding = self.kernel_size[::-1] - 1

        z_length, y_length, x_length = size + base_padding

        # test if the x-length is not perfect, pad accordingly
        if not (
            (x_length // self.stride[0] - 1) * self.stride[0] + self.kernel_size[0]
            == x_length
        ):
            desired_chunks_x = math.ceil(x_length / self.stride[0])
            x_length = (desired_chunks_x - 1) * self.stride[0] + self.kernel_size[0]

        if not (
            (y_length // self.stride[1] - 1) * self.stride[1] + self.kernel_size[1]
            == y_length
        ):
            desired_chunks_y = math.ceil(y_length / self.stride[1])
            y_length = (desired_chunks_y - 1) * self.stride[1] + self.kernel_size[1]

        if not (
            (z_length // self.stride[2] - 1) * self.stride[2] + self.kernel_size[2]
            == z_length
        ):
            desired_chunks_z = math.ceil(z_length / self.stride[2])
            z_length = (desired_chunks_z - 1) * self.stride[2] + self.kernel_size[2]

        self.processed_shape = (z_length, y_length, x_length, num_channel)
        self.position = (
            # x-axis, y-axis, z-axis
            slice(base_padding[2], base_padding[2] + size[2]),
            slice(base_padding[1], base_padding[1] + size[1]),
            slice(base_padding[0], base_padding[0] + size[0]),
        )

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Example:
            array:      [
                         [[A1 B1 C1 D1]
                          [E1 F1 G1 H1]
                          [I1 J1 K1 L1]
                          [M1 N1 O1 P1]]
                         [[A2 B2 C2 D2]
                          [E2 F2 G2 H2]
                          [I2 J2 K2 L2]
                          [M2 N2 O2 P2]]
                         [[A3 B3 C3 D3]
                          [E3 F3 G3 H3]
                          [I3 J3 K3 L3]
                          [M3 N3 O3 P3]]
                         [[A4 B4 C4 D4]
                          [E4 F4 G4 H4]
                          [I4 J4 K4 L4]
                          [M4 N4 O4 P4]]
                        ]

            kernel:     [
                         [[1 2 3]
                          [4 5 6]
                          [7 8 9]]
                         [[a b c]
                          [d e f]
                          [h i j]]
                         [[k l m]
                          [n o p]
                          [q r s]]
                        ]
            stride:     (1, 1, 1)
            processed:  [
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0 A1 B1 C1 D1  0  0]
                          [ 0  0 E1 F1 G1 H1  0  0]
                          [ 0  0 I1 J1 K1 L1  0  0]
                          [ 0  0 M1 N1 O1 P1  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0 A2 B2 C2 D2  0  0]
                          [ 0  0 E2 F2 G2 H2  0  0]
                          [ 0  0 I2 J2 K2 L2  0  0]
                          [ 0  0 M2 N2 O2 P2  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0 A3 B3 C3 D3  0  0]
                          [ 0  0 E3 F3 G3 H3  0  0]
                          [ 0  0 I3 J3 K3 L3  0  0]
                          [ 0  0 M3 N3 O3 P3  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0 A4 B4 C4 D4  0  0]
                          [ 0  0 E4 F4 G4 H4  0  0]
                          [ 0  0 I4 J4 K4 L4  0  0]
                          [ 0  0 M4 N4 O4 P4  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]
                          [ 0  0  0  0  0  0  0  0]]
                        ]

        :param raw_data: unprocessed 4D tensor (#z, #y, #x, #channel)
        :return: 0 padded array
        """

        processed = np.zeros(self.processed_shape)
        x_slice, y_slice, z_slice = self.position
        # all channels
        processed[z_slice, y_slice, x_slice, :] = raw_data
        return processed

    slices = slices3d

    unpad = unpad3d


class Same(Padding):
    """
    The same padding, also called half padding, pads
    0 around the input evenly so that output size will
    be math.ceil(input_size / stride).

    For same paddings with stride of 1, the output size
    is the same as the input size.
    """


class Same1D(Same):
    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        :param raw_data: unprocessed data
        :return: None
        """

        length, num_channel = raw_data.shape

        output_length = math.ceil(length / self.stride)
        desired_len = (output_length - 1) * self.stride + self.kernel_size

        # see Same2D
        desired_len = max(output_length, desired_len)

        num_pad = desired_len - length
        # left floor right ceil
        left_padding = num_pad // 2

        self.processed_shape = (desired_len, num_channel)
        self.position = (slice(left_padding, left_padding + length),)

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Examples:
            array:      [A B C D E F G]
            kernel:     [1 2 3]
            stride:     1
            processed:  [0 0 A B C D E F G 0 0]

            array:      [A B C D E F G H]
            kernel:     [1 2 3]
            stride:     2
            processed:  [0 0 A B C D E F G H 0]

        :param raw_data: unprocessed 2D tensor (time, #channel)
        :return: 0 padded array
        """
        processed = np.zeros(self.processed_shape)
        # all channels
        (x_slice,) = self.position
        processed[x_slice, :] = raw_data
        return processed

    slices = slices1d

    unpad = unpad1d


class Same2D(Padding):
    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        :param raw_data: unprocessed data (3D tensor)
        :return: None
        """

        *length, num_channel = raw_data.shape

        # since no unpacking is needed,
        # length shape -> (#x, #y)
        # whereas in full it's (#y, #x)
        # when modify pay caution on this
        length = np.array(length[::-1])

        output_length = np.ceil(length / self.stride).astype(int)
        desired_len = (output_length - 1) * self.stride + self.kernel_size

        # if component of output_length == 1 and kernel_size < length
        # then desired_len will be smaller than original array
        # i.e [[6] [6] [6]] with stride 3 and kernel_size of 2
        # will result in desired_len be 2, which is smaller than 3
        # thus causing num_pad to be negative. clamp the value to avoid issue
        desired_len = np.maximum(desired_len, length)

        num_pad = desired_len - length
        # left floor right ceil
        # star_padding.shape -> (#x, #y)
        start_padding = num_pad // 2

        # after all operation done in (#x, #y), reverse it back to (#y, #x)
        self.processed_shape = (*desired_len[::-1], num_channel)
        self.position = (
            # x-axis, y-axis
            slice(start_padding[0], start_padding[0] + length[0]),
            slice(start_padding[1], start_padding[1] + length[1]),
        )

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Examples:
            array:      [[A B C D]
                         [E F G H]
                         [I J K L]
                         [M N O P]]
            kernel:     [[1 2 3]
                         [4 5 6]
                         [7 8 9]]
            stride:     (1, 1)
            processed:  [[0 0 0 0 0 0]
                         [0 A B C D 0]
                         [0 E F G H 0]
                         [0 I J K L 0]
                         [0 M N O P 0]
                         [0 0 0 0 0 0]]

            array:      [[A B C D]
                         [E F G H]
                         [I J K L]
                         [M N O P]]
            kernel:     [[[1 2 3]
                         [4 5 6]
                         [7 8 9]]
            stride:     (2, 3)
            processed:  [[0 0 0 0 0]
                         [0 A B C D]
                         [0 E F G H]
                         [0 I J K L]
                         [0 M N O P]
                         [0 0 0 0 0]]

        :param raw_data: unprocessed 3D tensor (#y, #x, #channel)
        :return: 0 padded array
        """

        processed = np.zeros(self.processed_shape)
        x_slice, y_slice = self.position
        # all channels
        processed[y_slice, x_slice, :] = raw_data
        return processed

    slices = slices2d

    unpad = unpad2d


class Same3D(Padding):
    def build(self, raw_data):
        """
        Adds a `processed_shape` attribute for creating zero
        array with correct shape, as well as `position` indicating
        where to put the original array in the new array.

        :param raw_data: unprocessed data (4D tensor)
        :return: None
        """

        *length, num_channel = raw_data.shape

        # see same2d for caution
        # length.shape -> (#x, #y, #z)
        length = np.array(length[::-1])

        output_length = np.ceil(length / self.stride).astype(int)
        desired_len = (output_length - 1) * self.stride + self.kernel_size
        desired_len = np.maximum(desired_len, length)

        num_pad = desired_len - length
        # left floor right ceil
        # star_padding.shape -> (#x, #y, #z)
        start_padding = num_pad // 2

        self.processed_shape = (*desired_len[::-1], num_channel)
        self.position = (
            # x-axis, y-axis, z-axis
            slice(start_padding[0], start_padding[0] + length[0]),
            slice(start_padding[1], start_padding[1] + length[1]),
            slice(start_padding[2], start_padding[2] + length[2]),
        )

        super().build(raw_data)

    def pad(self, raw_data):
        """
        Example:
            array:      [
                         [[A1 B1 C1 D1]
                          [E1 F1 G1 H1]
                          [I1 J1 K1 L1]
                          [M1 N1 O1 P1]]
                         [[A2 B2 C2 D2]
                          [E2 F2 G2 H2]
                          [I2 J2 K2 L2]
                          [M2 N2 O2 P2]]
                         [[A3 B3 C3 D3]
                          [E3 F3 G3 H3]
                          [I3 J3 K3 L3]
                          [M3 N3 O3 P3]]
                         [[A4 B4 C4 D4]
                          [E4 F4 G4 H4]
                          [I4 J4 K4 L4]
                          [M4 N4 O4 P4]]
                        ]

            kernel:     [
                         [[1 2 3]
                          [4 5 6]
                          [7 8 9]]
                         [[a b c]
                          [d e f]
                          [h i j]]
                         [[k l m]
                          [n o p]
                          [q r s]]
                        ]
            stride:     (1, 1, 1)
            processed:  [[[ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0]
                          [ 0 A1 B1 C1 D1  0]
                          [ 0 E1 F1 G1 H1  0]
                          [ 0 I1 J1 K1 L1  0]
                          [ 0 M1 N1 O1 P1  0]
                          [ 0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0]
                          [ 0 A2 B2 C2 D2  0]
                          [ 0 E2 F2 G2 H2  0]
                          [ 0 I2 J2 K2 L2  0]
                          [ 0 M2 N2 O2 P2  0]
                          [ 0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0]
                          [ 0 A3 B3 C3 D3  0]
                          [ 0 E3 F3 G3 H3  0]
                          [ 0 I3 J3 K3 L3  0]
                          [ 0 M3 N3 O3 P3  0]
                          [ 0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0]
                          [ 0 A4 B4 C4 D4  0]
                          [ 0 E4 F4 G4 H4  0]
                          [ 0 I4 J4 K4 L4  0]
                          [ 0 M4 N4 O4 P4  0]
                          [ 0  0  0  0  0  0]]
                         [[ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]
                          [ 0  0  0  0  0  0]]]

        :param raw_data: unprocessed 4D tensor (#z, #y, #x, #channel)
        :return: 0 padded array
        """

        processed = np.zeros(self.processed_shape)
        x_slice, y_slice, z_slice = self.position
        # all channels
        processed[z_slice, y_slice, x_slice, :] = raw_data
        return processed

    slices = slices3d

    unpad = unpad3d


class Valid(Padding):
    """No padding, same as 'valid' in keras"""

    def build(self, raw_data):
        self.processed_shape = raw_data.shape
        # original position
        self.position = tuple(slice(None) for _ in raw_data.shape)
        super().build(raw_data)

    def pad(self, raw_data):
        # self.input = self.output = data
        return raw_data


class Valid1D(Valid):
    slices = slices1d

    unpad = unpad1d


class Valid2D(Valid):
    slices = slices2d

    unpad = unpad2d


class Valid3D(Valid):
    slices = slices3d

    unpad = unpad3d
