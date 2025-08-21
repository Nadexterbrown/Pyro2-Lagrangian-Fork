
"""
Proxy to Pyro's compressible unsplit flux machinery while forcing the
Riemann solver to this package's Lagrangian/ALE implementation.
All functions are re-exported from pyro.compressible.unsplit_fluxes.
"""
from pyro.compressible import unsplit_fluxes as _base
from . import riemann as riemann  # our riemann module

# Monkey-patch: ensure base uses our riemann module
_base.riemann = riemann

# Re-export everything public from base
__all__ = [n for n in dir(_base) if not n.startswith("_")]
globals().update({n: getattr(_base, n) for n in __all__})
