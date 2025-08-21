
import importlib

# We inherit Pyro's compressible Simulation to reuse mesh, state storage,
# BC handling, unsplit update, IO, and visualization surfaces.
from pyro.compressible.simulation import Simulation as EulerSimulation
from pyro.compressible import unsplit_fluxes as _base_unsplit

# Our Lagrangian/ALE Riemann module (same public names as Pyro's riemann)
from . import riemann as _lagr_riemann

class Simulation(EulerSimulation):
    """
    Drop-in replacement for Pyro's compressible.Simulation that only swaps the
    Riemann solver with a Lagrangian/ALE variant. API and behavior from the
    user's perspective (runtime params, problem registration, getters, IO)
    are identical to Pyro's compressible solver so it can be driven by
    pyro_sim.Pyro without changes.
    """
    name = "compressible_lagrangian"

    # IMPORTANT: match pyro_sim.Pyro() call signature exactly
    def __init__(self, solver_name, problem_name, problem_func, rp,
                 problem_finalize_func=None, problem_source_func=None, timers=None):
        # Call the base class constructor with the same arguments that pyro_sim passes
        super().__init__(solver_name, problem_name, problem_func, rp,
                         problem_finalize_func=problem_finalize_func,
                         problem_source_func=problem_source_func,
                         timers=timers)

        # Select our Lagrangian/ALE riemann implementation for the entire solver.
        # We do this once at construction; the unsplit flux machinery in Pyro will
        # import and call riemann.* by name, now bound to our module.
        _base_unsplit.riemann = _lagr_riemann

        # Keep honoring the user's riemann choice (e.g., 'HLLC' or 'CGF')
        # via the runtime parameter 'compressible.riemann'. We simply ensure
        # that the function resolved comes from our module instead of Pyro's.
        # Nothing else changes for the user.

    # No further overrides are needed: evolve(), compute_timestep(), do_output(),
    # write(), dovis(), finished(), finalize() are inherited from the Eulerian
    # compressible solver and continue to work unchanged.
