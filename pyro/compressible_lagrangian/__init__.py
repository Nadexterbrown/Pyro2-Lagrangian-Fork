
"""
Lagrangian (ALE) variant of Pyro2's compressible solver with 1:1 module layout.
We reuse Pyro's Eulerian machinery wherever possible and only replace the
Riemann solver with an ALE/Lagrangian version.

Modules:
  - eos, reconstruction, interface, derives: re-export Pyro's modules
  - unsplit_fluxes: proxy to Pyro's unsplit flux module, patched to use our riemann
  - riemann: Pyro-compatible API, returns ALE fluxes (F - w_n U_face), w_n = S_M
  - simulation: subclass/wrapper of Pyro compressible Simulation that installs the patch
"""
__all__ = ["simulation"]

from .simulation import (Simulation)
