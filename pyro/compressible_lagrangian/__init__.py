
"""Purely Lagrangian (no ALE, no remap) 2-D compressible solver for Pyro2.

Exposes both the `simulation` submodule and the `Simulation` class to match
Pyro's import/usage pattern.
"""
from . import simulation as simulation
from .simulation import Simulation
__all__ = ["simulation", "Simulation"]
