from __future__ import annotations
import numpy as np

class CellState:
    def __init__(self, mesh, rp):
        ny, nx = mesh.ny, mesh.nx
        self.gamma = float(rp.get_param("eos.gamma"))
        # conserved (per cell)
        self.m      = np.zeros((ny, nx))
        self.rho_u  = np.zeros((ny, nx))
        self.rho_v  = np.zeros((ny, nx))
        self.rho_E  = np.zeros((ny, nx))
        # derived
        self.rho    = np.zeros((ny, nx))
        self.u      = np.zeros((ny, nx))
        self.v      = np.zeros((ny, nx))
        self.p      = np.zeros((ny, nx))

    def sync_primitives(self, mesh):
        # density from constant mass / current area
        mesh_area = mesh.area
        self.rho = np.where(mesh_area > 0.0, self.m / mesh_area, 0.0)
        # velocity
        rh = np.where(self.rho > 1e-30, self.rho, 1.0)
        self.u = self.rho_u / rh
        self.v = self.rho_v / rh
        # pressure from total energy
        ke = 0.5 * self.rho * (self.u*self.u + self.v*self.v)
        eint = np.maximum(self.rho_E - ke, 0.0)
        self.p = np.maximum((self.gamma - 1.0) * eint, 1e-30)
