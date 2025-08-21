
from pyro.compressible.simulation import Simulation as EulerSimulation
from pyro.compressible import unsplit_fluxes as _base_unsplit
from . import riemann as _lagr_riemann

class Simulation(EulerSimulation):
    name = "compressible_lagrangian"

    def __init__(self, solver_name, problem_name, problem_func, rp,
                 problem_finalize_func=None, problem_source_func=None, timers=None):
        super().__init__(solver_name, problem_name, problem_func, rp,
                         problem_finalize_func=problem_finalize_func,
                         problem_source_func=problem_source_func,
                         timers=timers)
        # Route all unsplit calls to our riemann module (API-compatible wrapper)
        _base_unsplit.riemann = _lagr_riemann
