
import importlib
from pyro.compressible.simulation import Simulation as EulerSimulation
from pyro.compressible import unsplit_fluxes as _base_unsplit
from . import riemann as _lagr_riemann

class Simulation(EulerSimulation):
    """
    Drop-in Simulation identical to Pyro's compressible solver except that
    the Riemann solver used by the unsplit flux machinery is the Lagrangian/ALE
    one in this package.
    """
    name = "compressible_lagrangian"

    def __init__(self, rp):
        # initialize base compressible solver (allocations, runtime params, etc.)
        super().__init__(rp)

        # Force the unsplit flux module to use our riemann
        _base_unsplit.riemann = _lagr_riemann

        # Respect user's riemann selection (HLLC/CGF) through our dispatcher
        # (Pyro stores it under "compressible.riemann" typically)
        try:
            rsel = self.rp.get_param("compressible.riemann")
        except Exception:
            rsel = getattr(self, "riemann", "HLLC")
        setattr(self, "riemann", rsel or "HLLC")
