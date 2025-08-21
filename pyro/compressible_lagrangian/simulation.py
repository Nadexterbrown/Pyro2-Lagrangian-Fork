
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

from .mesh import MovingQuadMesh
from .state import LagrangianState, eos_pressure, cons_from_prim, prim_from_cons
from .time_integration import SSPRK2Stepper
from .forces import accumulate_pressure_forces_and_work
from .riemann import face_states_and_star
from .reconstruction import muscl_reconstruct
from .viscosity import vn_r_viscosity, HourglassControl
from .boundary import BoundaryManager

# Minimal cc_data shim to look like Pyro's accessors
class CCDataShim:
    def __init__(self, state: LagrangianState):
        self._state = state
        self.grid = state.mesh  # for .ilo etc parity on usage sites
        self.t = 0.0
    def get_var(self, name: str):
        if name == "density":
            return self._state.rho
        if name == "x-momentum":
            return self._state.rho * self._state.u[..., 0]
        if name == "y-momentum":
            return self._state.rho * self._state.u[..., 1]
        if name == "energy":
            return self._state.rho * self._state.E
        if name == "pressure":
            return eos_pressure(self._state.gamma, self._state.rho, self._state.p_eint())
        raise KeyError(name)

@dataclass
class Timers:
    pass

class Simulation:
    """Pyro2-compatible driver for a purely Lagrangian compressible solver.

    This class mirrors the constructor signature and hooks expected by
    `pyro.pyro_sim` so it can be used as a drop-in solver.
    """
    name = "compressible_lagrangian_pure"

    def __init__(self, solver_name, problem_name, problem_func, rp,
                 problem_finalize_func=None, problem_source_func=None, timers=None):
        self.solver_name = solver_name
        self.problem_name = problem_name
        self.problem_func = problem_func
        self.problem_finalize_func = problem_finalize_func
        self.problem_source_func = problem_source_func
        self.rp = rp
        self.timers = timers or Timers()

        # Runtime params
        self.gamma = float(self.rp.get_param("eos.gamma", 1.4))
        nx = int(self.rp.get_param("mesh.nx", 64))
        ny = int(self.rp.get_param("mesh.ny", 64))
        xmin = float(self.rp.get_param("mesh.xmin", 0.0))
        xmax = float(self.rp.get_param("mesh.xmax", 1.0))
        ymin = float(self.rp.get_param("mesh.ymin", 0.0))
        ymax = float(self.rp.get_param("mesh.ymax", 1.0))

        self.mesh = MovingQuadMesh(nx, ny, xmin, xmax, ymin, ymax)
        self.state = LagrangianState(self.mesh, self.gamma)
        self.cc_data = CCDataShim(self.state)

        # Controls
        self.cfl = float(self.rp.get_param("driver.cfl", 0.5))
        self.max_steps = int(self.rp.get_param("driver.max_steps", 10000))
        self.tmax = float(self.rp.get_param("driver.tmax", 1.0))

        # Stabilization toggles (off by default)
        self.visc_coeff = float(self.rp.get_param("lagrangian.visc_coeff", 0.0))
        self.hg_coeff = float(self.rp.get_param("lagrangian.hg_coeff", 0.0))
        self.hg = HourglassControl(self.hg_coeff)

        # Boundaries
        self.bcs = BoundaryManager(self.rp)

        # Time integrator
        self.stepper = SSPRK2Stepper()

        # bookkeeping
        self.nstep = 0
        self.t = 0.0

    # ---- API hooks expected by pyro_sim.py ----
    def initialize(self):
        # Delegate to problem init to fill density/momentum/energy
        self.problem_func(self.state.as_pyro_cc_like(), self.rp)
        # Initialize per-cell mass from density and area
        self.state.initialize_cell_mass_from_density()

    def preevolve(self):
        pass

    def compute_timestep(self) -> float:
        # dt <= CFL * min(ell / a)
        a = np.sqrt(self.gamma * np.maximum(self.state.p(), 0.0) / np.maximum(self.state.rho, 1e-30))
        ell = self.mesh.inscribed_diameter()
        dt_loc = np.where(a > 0.0, ell / a, np.inf)
        dt = self.cfl * float(np.min(dt_loc))
        return dt

    # Backwards-compat alias used by some Pyro versions
    dtdrive = compute_timestep

    def evolve(self, dt: float):
        """One SSP-RK2 step of a purely Lagrangian update."""
        def rhs():
            # 1) Reconstruct primitives at faces at half-step (Hancock predictor)
            prim = self.state.primitive_tuple()
            grad = muscl_reconstruct(self.mesh, prim)
            faces = face_states_and_star(self.mesh, self.gamma, prim, grad)

            # Optional stabilization
            if self.visc_coeff > 0.0:
                vn_r_viscosity(self.mesh, faces, self.state, self.visc_coeff)
            if self.hg.coeff > 0.0:
                self.hg.apply(self.mesh, self.state)

            # 2) Pressure forces & work using p* and face normal speed u*_n
            mom_rhs, ener_rhs = accumulate_pressure_forces_and_work(self.mesh, faces)

            return mom_rhs, ener_rhs, faces

        # SSP-RK2 for conservative variables (momentum, energy) in Lagrangian form
        mom0 = (self.state.rho[..., None] * self.state.u).copy()
        Et0 = (self.state.rho * self.state.E).copy()

        # Stage 1
        mom_rhs, ener_rhs, faces = rhs()
        mom1 = mom0 + dt * mom_rhs
        Et1  = Et0  + dt * ener_rhs
        self.state.set_cons(mom1, Et1)

        # Move mesh using face velocities from stage-1 star states
        self.mesh.move_nodes(faces, dt)

        # Update density from constant mass / new volumes
        self.state.update_density_from_mass()

        # Stage 2
        mom_rhs, ener_rhs, faces = rhs()
        mom2 = 0.5 * (mom0 + (mom1 + dt * mom_rhs))
        Et2  = 0.5 * (Et0  + (Et1  + dt * ener_rhs))
        self.state.set_cons(mom2, Et2)

        # Move mesh again for second stage (Heun)
        self.mesh.move_nodes(faces, dt)

        # Update density from mass
        self.state.update_density_from_mass()

        # Apply boundary conditions (moving walls, outflow, periodic)
        self.bcs.apply(self.mesh, self.state, self.t + dt)

        # Bookkeeping
        self.t += dt
        self.cc_data.t = self.t

    def finalize(self):
        if self.problem_finalize_func:
            self.problem_finalize_func()

    # Utilities used by pyro_sim driver
    def get_var(self, name: str):
        return self.cc_data.get_var(name)

    # Simple drive loop (not used by pyro_sim, but handy)
    def timestep(self):
        dt = self.compute_timestep()
        # clip to tmax
        if self.t + dt > self.tmax:
            dt = self.tmax - self.t
        return dt
