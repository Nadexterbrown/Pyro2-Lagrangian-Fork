from __future__ import annotations
import numpy as np
from .util import rp_get

class CellState:
    def __init__(self, mesh, rp):
        ny, nx = mesh.ny, mesh.nx
        self.gamma = float(rp_get(rp, "eos.gamma", 1.4))
        self.m      = np.zeros((ny, nx))
        self.rho_u  = np.zeros((ny, nx))
        self.rho_v  = np.zeros((ny, nx))
        self.rho_E  = np.zeros((ny, nx))
        self.rho    = np.zeros((ny, nx))
        self.u      = np.zeros((ny, nx))
        self.v      = np.zeros((ny, nx))
        self.p      = np.zeros((ny, nx))

    def sync_primitives(self, mesh):
        area = mesh.area
        self.rho = np.where(area > 0.0, self.m / area, 0.0)
        rh = np.where(self.rho > 1e-30, self.rho, 1.0)
        self.u = self.rho_u / rh
        self.v = self.rho_v / rh
        ke = 0.5 * self.rho * (self.u*self.u + self.v*self.v)
        eint = np.maximum(self.rho_E - ke, 0.0)
        self.p = np.maximum((self.gamma - 1.0) * eint, 1e-30)
