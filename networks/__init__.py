import numpy as np

# import util pack first, since it does not rely on other packs
# some content in util should not be exposed
# i.e. assertions
from .util import batches
from .util.history import History, plot_metrics
from .util.dataset_utils import index_to_onehot

# import core structures
# Dependency of core structures:
#
# optimizers (no dependency)
#
# activations -> layers -> util
#                   losses _↑

from .optimizers import Optimizer, SGD, Adam
from .layers import (
    Dense,
    Input,
    Layer,
    PreprocessingLayer,
    Flatten,
    Reshape,
    Activation,
    ReLU,
    LeakyReLU,
    SoftMax,
    Sigmoid,
    Tanh,
    Conv1D
)
from .losses import Loss, CrossEntropy, MSE

# import engines after all other components load
from .engines import Sequential

from rich.traceback import install

np.seterr(all="raise")

install(show_locals=True)
