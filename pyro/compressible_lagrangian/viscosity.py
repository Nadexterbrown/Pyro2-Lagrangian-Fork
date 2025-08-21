
from __future__ import annotations
import numpy as np

def vn_r_viscosity(mesh, faces, state, Cq):
    """von Neumann-Richtmyer quadratic viscosity on compressive faces."""
    div = ((faces["u_vec_e"][...,0] - faces["u_vec_w"][...,0]) +
           (faces["u_vec_n"][...,1] - faces["u_vec_s"][...,1]))
    mask = div < 0.0
    h = mesh.inscribed_diameter()
    q = np.zeros_like(state.rho)
    q[mask] = Cq * state.rho[mask] * (0.5*h[mask]*div[mask])**2
    for key in ["pstar_w","pstar_e","pstar_s","pstar_n"]:
        faces[key] += q

class HourglassControl:
    def __init__(self, coeff=0.0):
        self.coeff = coeff
    def apply(self, mesh, state):
        if self.coeff <= 0.0: return
        # Placeholder for subcell stabilization
        pass
