
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def cons_from_prim(gamma, rho, u, v, p):
    Eint = p / ((gamma-1.0)*rho + 1e-30)
    E = Eint + 0.5*(u*u + v*v)
    mom = rho[..., None] * np.stack([u, v], axis=-1)
    return mom, rho*E

def prim_from_cons(gamma, rho, mom, Et):
    u = mom[...,0]/np.maximum(rho,1e-30)
    v = mom[...,1]/np.maximum(rho,1e-30)
    E = Et/np.maximum(rho,1e-30)
    ke = 0.5*(u*u+v*v)
    p = (gamma-1.0)*rho*np.maximum(E-ke,0.0)
    return u, v, p, E

def eos_pressure(gamma, rho, eint):
    return (gamma-1.0)*rho*eint

@dataclass
class LagrangianState:
    mesh: any
    gamma: float

    def __post_init__(self):
        ny, nx = self.mesh.ny, self.mesh.nx
        self.rho = np.ones((ny, nx))
        self.u = np.zeros((ny, nx, 2))
        self.E = np.ones((ny, nx))  # specific total energy
        self.m = np.ones((ny, nx))  # cell mass (constant)
    def as_pyro_cc_like(self):
        # Provide just enough for a Pyro problem init function
        class CC:
            def __init__(self,_self):
                self._s=_self
            def get_var(self, name):
                if name=="density": return self._s.rho
                if name=="x-momentum": return self._s.rho*self._s.u[...,0]
                if name=="y-momentum": return self._s.rho*self._s.u[...,1]
                if name=="energy": return self._s.rho*self._s.E
                raise KeyError(name)
        return CC(self)
    def initialize_cell_mass_from_density(self):
        A = self.mesh.cell_area()
        self.m = self.rho * A
    def update_density_from_mass(self):
        A = self.mesh.cell_area()
        self.rho = self.m / np.maximum(A, 1e-30)
    def p(self):
        u = self.u[...,0]; v=self.u[...,1]
        ke = 0.5*(u*u+v*v)
        return (self.gamma-1.0)*self.rho*np.maximum(self.E - ke, 0.0)
    def p_eint(self):
        u = self.u[...,0]; v=self.u[...,1]
        ke = 0.5*(u*u+v*v)
        return np.maximum(self.E - ke, 0.0)
    def primitive_tuple(self):
        u = self.u[...,0]; v=self.u[...,1]
        return (self.rho.copy(), u.copy(), v.copy(), self.p().copy())
    def set_cons(self, mom, Et):
        self.u[...,0] = mom[...,0]/np.maximum(self.rho,1e-30)
        self.u[...,1] = mom[...,1]/np.maximum(self.rho,1e-30)
        self.E = Et / np.maximum(self.rho,1e-30)
