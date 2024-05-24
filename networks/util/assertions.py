import functools

import numpy as np

from .ctx_managers import ChainError


def assert_not_empty(seq, errmsg):
    """
    Checks whether the sequence is empty, if so error is raised

    :param seq: sequence of value
    :param errmsg: error message
    :return: original sequence
    """
    if not seq:
        raise TypeError(errmsg) from None
    return seq


def assert_scaler(val, errmsg):
    """
    Checks whether val can be treated as a float.
    Raises TypeError if unable to convert.

    :param val: value to be converted
    :param errmsg: error message
    :return: float(value)
    """
    try:
        return float(val)
    except TypeError:
        raise TypeError(errmsg) from None


def assert_isinstance(obj, cls, errmsg):
    """
    Checks whether an object `obj` is an instance of the class `cls`,
    if no, raise exception with the template

    :param obj: an object
    :param cls: class or tuple
    :param errmsg: string of error message
    :return: original object or no return
    """
    if not isinstance(obj, cls):
        raise TypeError(errmsg)
    return obj


def assert_notinstance(obj, cls, errmsg):
    """
    Performs a reverse check to `assert_isinstance`

    :param obj: an object
    :param cls: class or tuple
    :param errmsg: error message
    :return: original object or no return
    """
    if isinstance(obj, cls):
        raise TypeError(errmsg)
    return obj


def assert_max_ndim(arr, max_dim, errmsg):
    """
    Raises an error if array `arr` has a dimension over `max_dim`

    :param arr: ndarray
    :param max_dim: max dimension allowed
    :param errmsg: error message
    :return: original array or no return
    """
    if arr.ndim > max_dim:
        raise TypeError(errmsg)
    return arr


def assert_ndim(arr, ndim, errmsg):
    """
    Raises an error if array `arr` has a dimension over `max_dim`

    :param arr: ndarray
    :param ndim: dimension allowed
    :param errmsg: error message
    :return: original array or no return
    """
    if arr.ndim != ndim:
        raise TypeError(errmsg)
    return arr


def assert_same_length(x, y, errmsg):
    """
    Checks if two sequences have the same length

    :param x: x
    :param y: y
    :param errmsg: error message
    :return: None
    """
    if len(x) != len(y):
        raise TypeError(errmsg)


def assert_reshape_compatibility(shape1, shape2, errmsg):
    """
    If two shapes are compatible, then two products of their shapes should be the same.
    Note all shapes should consist positive integers, -1 is not allowed.

    :param shape1: input shape
    :param shape2: output shape
    :param errmsg: error message
    :return: None
    """
    if np.prod(shape1) != np.prod(shape2):
        raise TypeError(errmsg)


def assert_all_positive(arr, errmsg):
    """
    Checks whether array `arr` has all positive elements

    :param arr:
    :param errmsg:
    :return: original array
    """
    if not np.all(arr > 0):
        raise TypeError(errmsg)
    return arr


def assert_integer_array(arr, errmsg):
    """
    Checks whether array `arr` has `dtype` of integer

    :param arr: array / scaler to be examined
    :param errmsg: error message
    :return: original array
    """
    if not np.issubdtype(arr.dtype, np.integer) and not isinstance(arr, int):
        raise TypeError(errmsg)
    return arr


def assert_positive_int(val, errmsg):
    if not isinstance(val, (np.integer, int)):
        raise TypeError(errmsg)
    if val <= 0:
        raise TypeError(errmsg)
    return val


def assert_built(self, errmsg):
    if not self.built:
        raise TypeError(errmsg)


def requires_build(errmsg_or_func):
    if not isinstance(errmsg_or_func, str):
        @functools.wraps(errmsg_or_func)
        def inner(self, *args, **kwargs):
            assert_built(self, "Model not built, please build the model first")
            return errmsg_or_func(self, *args, **kwargs)

        return inner

    def decorator(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            assert_built(self, errmsg_or_func)
            return func(self, *args, **kwargs)

        return inner

    return decorator


def requires_build_padding(func):
    return requires_build("Padding not built. Please build this padding first")(func)


def requires_layer_build(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        assert_built(self, f"{self.__class__.__qualname__} object {self} is not built. Please build this layer first")
        return func(self, *args, **kwargs)
    return inner


def assert_str(self, errmsg):
    if not isinstance(self, str):
        raise TypeError(errmsg)


def assert_length(x, n, errmsg):
    if len(x) != n:
        raise TypeError(errmsg)


def assert_valid_conv_attribute(x, n, errmsg, chained_errmsg):
    if isinstance(x, (np.integer, int)):
        with ChainError(TypeError(chained_errmsg)):
            assert_positive_int(x, errmsg)
            return np.array([x] * n)

    x = np.asarray(x)
    with ChainError(TypeError(chained_errmsg)):
        assert_all_positive(x, errmsg)
        assert_integer_array(x, errmsg)
        assert_length(x, n, errmsg)
    return x


def assert_valid_stride(x, n: int, errmsg):
    """
    Valid stride can either be a positive integer representing
    same stride on all the axes, or it can be an n dimensional
    positive array, where n equals to dim

    :param x: object to be examined
    :param n: dimension of stride (int)
    :param errmsg: error message
    :return: n dimensional stride array
    """

    if n == 1:
        return assert_positive_int(x, errmsg)

    chained_errmsg = f"""Valid stride can either be:
    · A positive integer (same stride on all the axes)
    · A(n) {n}-dimensional positive array
{x} (object of {type(x)})
 is not a valid {n}-dimensional stride."""

    return assert_valid_conv_attribute(x, n, errmsg, chained_errmsg)


def assert_valid_kernel_size(x, n: int, errmsg):
    """
    Valid kernel can either be a positive integer representing
    same size on all the axes, or it can be an n dimensional
    positive array, where n equals to dim

    :param x: object to be examined
    :param n: dimension of kernel (int)
    :param errmsg: error message
    :return: n dimensional kernel array
    """

    if n == 1:
        return assert_positive_int(x, errmsg)

    chained_errmsg = f"""Valid kernel size can either be:
    · A positive integer (same size on all the axes)
    · A(n) {n}-dimensional positive array
{x} (object of {type(x)})
 is not a valid {n}-dimensional kernel."""

    return assert_valid_conv_attribute(x, n, errmsg, chained_errmsg)


def assert_valid_output_shape(arr):
    """
    Performs a quick check on whether `arr` is positive, 0/1d, and integer.

    1. converts `arr` into ndarray,
    2. Do the tests
    3. Return the converted array / raise exception

    :param arr: array / scaler to be examined
    :return: original array
    """
    arr = np.asarray(arr)
    try:
        assert_max_ndim(arr, 1, "")
        assert_integer_array(arr, "")
        assert_all_positive(arr, "")
    except TypeError:
        raise TypeError(f"""Invalid output shape: {arr}
Troubleshoot, check if array:
 - is a 1D vector or 0D scaler (array.shape: {arr.shape})
 - is made up by integer values (array.dtype: {arr.dtype})
 - contains all positive values (all positive? {np.all(arr > 0)})""") from None
    return arr
