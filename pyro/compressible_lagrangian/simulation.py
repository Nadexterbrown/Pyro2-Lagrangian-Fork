from __future__ import annotations
import numpy as np
from typing import Optional, Callable, Dict, Any

from .mesh import MovingQuadMesh, GridView
from .state import CellState
from .time_integration import SSPRK2Stepper
from .boundary import BoundaryManager

class CCDataShim:
    """Expose a Pyro-like cc_data interface over the moving mesh.
    This is enough for plotting/diagnostics that expect get_var(name) and .grid.
    """
    def __init__(self, mesh: MovingQuadMesh, state: CellState):
        self.grid: GridView = mesh.pyro_grid_view()
        self._state = state
        self.t = 0.0

    def get_var(self, name: str):
        if name == "density":
            return self._state.rho
        if name == "x-momentum":
            return self._state.rho_u
        if name == "y-momentum":
            return self._state.rho_v
        if name == "energy":
            return self._state.rho_E
        raise KeyError(name)

class Simulation:
    name = "compressible_lagrangian_pure"

    def __init__(self, solver_name, problem_name, problem_func, rp,
                 problem_finalize_func: Optional[Callable] = None,
                 problem_source_func: Optional[Callable] = None,
                 timers: Optional[Dict[str, Any]] = None):
        # Pyro2-compatible constructor signature
        self.rp = rp
        self.problem_name = problem_name
        self.problem_func = problem_func
        self.problem_finalize_func = problem_finalize_func
        self.problem_source_func = problem_source_func
        self.timers = timers or {}

        self.mesh: Optional[MovingQuadMesh] = None
        self.state: Optional[CellState] = None
        self.bc: Optional[BoundaryManager] = None
        self.stepper: Optional[SSPRK2Stepper] = None
        self.cc_data: Optional[CCDataShim] = None
        self.t: float = 0.0
        self.dt: Optional[float] = None

    # Hooks expected by pyro_sim.py
    def initialize(self):
        self.mesh = MovingQuadMesh(self.rp)
        self.state = CellState(self.mesh, self.rp)
        # delegate initialization of physical fields to problem module
        self.problem_func(self)
        self.state.sync_primitives(self.mesh)
        self.bc = BoundaryManager(self.rp)
        self.stepper = SSPRK2Stepper(self.rp)
        self.cc_data = CCDataShim(self.mesh, self.state)
        self.cc_data.t = 0.0

    def preevolve(self):
        # no-op hook for parity
        pass

    def compute_timestep(self):
        assert self.mesh is not None and self.state is not None
        self.dt = self.mesh.cfl_timestep(self.state, self.rp.get_param("driver.cfl"))
        return self.dt

    dtdrive = compute_timestep  # alias accepted by some drivers

    def evolve(self, dt: float):
        assert self.stepper and self.mesh and self.state and self.bc
        self.stepper.advance(self.mesh, self.state, self.bc, dt)
        self.state.sync_primitives(self.mesh)

    def timestep(self):
        dt = self.compute_timestep()
        self.evolve(dt)
        self.t += dt
        if self.cc_data: self.cc_data.t = self.t

    def finalize(self):
        if self.problem_finalize_func:
            self.problem_finalize_func(self)
