import numpy as np


def index_to_onehot(n_class, seq):
    """
    Converts index to one-hot vector

    i.e. index_to_onehot(6, [1, 2, 3])
         >>> np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
         ])

    :param n_class: number of class, length of vector generated
    :param seq: a sequance of indexes
    :return: ndarray of onehot vectors
    """
    ret = np.zeros((len(seq), n_class))
    for n_row, index in enumerate(seq):
        ret[n_row][index] = 1
    return ret
