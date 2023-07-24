import math

import numpy as np

from .assertions import assert_same_length, assert_positive_int


def batches(x, y, size):
    """
    Generator shuffles and splits training dataset into batches

    :param x: training inputs
    :param y: training answers
    :param size: batch_size
    :return: batch of training data
    """
    assert_same_length(x, y, f"Training input containing {len(x)} examples while labels containing {len(y)} examples")
    assert_positive_int(size, f"Batch size must be a positive integer, got {size} of type {size.__class__}")
    length = math.ceil(len(x) / size)
    return _batcher(x, y, size, length)


def _batcher(x, y, size, length):
    """
    Actual implementation of batching generator

    :param x: training inputs
    :param y: training answers
    :param size: batch_size
    :param length: total examples
    :return: x_example, y_example
    """

    while True:
        random_indexes = np.random.randint(0, length-1, (size,))
        yield x[random_indexes], y[random_indexes]
