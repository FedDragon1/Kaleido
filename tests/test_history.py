import unittest

import numpy as np

from networks import *


class TestHistory(unittest.TestCase):

    def test_plot(self):
        history = History("test")
        history.extend(np.linspace(0, 1))
        history.plot()
