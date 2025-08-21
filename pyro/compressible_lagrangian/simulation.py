"""
Simulation class for compressible_lagrangian.
Currently delegates to compressible.Simulation for API compatibility.
"""

from pyro.compressible.simulation import Simulation as EulerianSimulation
from . import riemann

class Simulation(EulerianSimulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite riemann solver with our wrapper
        import pyro.compressible.riemann as base_riemann
        base_riemann.riemann_flux = riemann.riemann_flux
