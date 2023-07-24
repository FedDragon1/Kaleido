import numpy as np


def assert_not_empty(seq, errmsg):
    """
    Checks whether the sequence is empty, if so error is raised

    :param seq: sequence of value
    :param errmsg: error message
    :return:
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
    :return: None or no return
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
    :return: None or no return
    """
    if isinstance(obj, cls):
        raise TypeError(errmsg)
    return obj


def assert_ndim(arr, max_dim, errmsg):
    """
    Raises an error if array `arr` has a dimension over `max_dim`

    :param arr: ndarray
    :param max_dim: max dimension allowed
    :param errmsg: error message
    :return: None or no return
    """
    if arr.ndim > max_dim:
        raise TypeError(errmsg)
    return arr


def assert_same_length(x, y, errmsg):
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
    :return: None
    """
    if not np.all(arr > 0):
        raise TypeError(errmsg)
    return arr


def assert_integer_array(arr, errmsg):
    """
    Checks whether array `arr` has `dtype` of integer

    :param arr: array / scaler to be examined
    :param errmsg: error message
    :return: None
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


def assert_valid_output_shape(arr):
    """
    Performs a quick check on whether `arr` is positive, 0/1d, and integer.

    1. converts `arr` into ndarray,
    2. Do the tests
    3. Return the converted array / raise exception

    :param arr: array / scaler to be examined
    :return: None
    """
    arr = np.asarray(arr)
    try:
        assert_ndim(arr, 1, "")
        assert_integer_array(arr, "")
        assert_all_positive(arr, "")
    except TypeError:
        raise TypeError(f"""Invalid output shape: {arr}
Troubleshoot, check if array:
 - is a 1D vector or 0D scaler (array.shape: {arr.shape})
 - is made up by integer values (array.dtype: {arr.dtype})
 - contains all positive values (all positive? {np.all(arr > 0)})""") from None
    return arr
