# import util pack first, since it does not rely on other packs
# some content in util should not be exposed
# i.e. assertions
from .util import batches
from .util.history import *
from .util.dataset_utils import *

# import core structures
# Dependency of core structures:
#
# optimizers (no dependency)
#
# activations -> layers -> util
#                   losses _↑

from .optimizers import *
from .layers import *
from .losses import *

# import engines after all other components load
from .engines import *

from rich.traceback import install

np.seterr(all='raise')

install(show_locals=True)
