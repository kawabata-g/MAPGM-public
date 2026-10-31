"""MAPGM fitted by ADMM, in batch and streaming form."""

from mapgm.methods.mapgm.model import MAPGM
from mapgm.methods.mapgm.stream import StreamMAPGM

__all__ = ["MAPGM", "StreamMAPGM"]
