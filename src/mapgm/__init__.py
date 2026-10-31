"""Multi-Aspect Poisson Graphical Model (MAPGM)."""

# Sizes the BLAS and numba thread pools; it has to run before numpy and numba
# are first imported.  See mapgm._threading.
from mapgm import _threading  # noqa: F401  isort: skip

from mapgm.datasets import ModeSummaries, TensorDataLoader
from mapgm.methods import MAPGM, StreamMAPGM

__all__ = ["MAPGM", "StreamMAPGM", "ModeSummaries", "TensorDataLoader"]
