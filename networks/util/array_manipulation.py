import numpy as np


def array_to_tuple(arr):
    """
    Numpy 0d-array cannot be cast into tuple directly.
    This utility function helps with doing that

    :param arr: 0d or 1d array
    :return: tuple of that array
    """
    return (arr[()],) if arr.ndim == 0 else tuple(arr.tolist())


def to_array_with_type(array_like, astype, errmsg):
    """
    First converts `array_like` into numpy array, then convert the array into type provided.
    If fails, raise a TypeError with errmsg provided.

    :param array_like: object to be converted into numpy array
    :param astype: type converting into
    :param errmsg: error message
    :return: converted numpy array or no return
    """
    try:
        # astype returns a new array, not mutable from outside
        array_like = np.array(array_like).astype(astype)
    except TypeError:
        raise TypeError(errmsg)
    return array_like