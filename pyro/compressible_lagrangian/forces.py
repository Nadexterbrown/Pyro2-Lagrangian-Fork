
from __future__ import annotations
import numpy as np

def accumulate_pressure_forces_and_work(mesh, faces):
    """Sum pressure forces and work for each cell."""
    ny, nx = mesh.ny, mesh.nx
    mom_rhs = np.zeros((ny, nx, 2))
    ener_rhs = np.zeros((ny, nx))

    (nw, ne, ns, nn), (Lw, Le, Ls, Ln) = mesh.face_geometry()

    n_w = np.dstack([-np.ones_like(Lw), np.zeros_like(Lw)])
    n_e = np.dstack([ np.ones_like(Le), np.zeros_like(Le)])
    n_s = np.dstack([ np.zeros_like(Ls),-np.ones_like(Ls)])
    n_n = np.dstack([ np.zeros_like(Ln), np.ones_like(Ln)])

    mom_rhs[:, :, :] += -faces["pstar_w"][..., None] * n_w * Lw[..., None]
    mom_rhs[:, :, :] += -faces["pstar_e"][..., None] * n_e * Le[..., None]
    mom_rhs[:, :, :] += -faces["pstar_s"][..., None] * n_s * Ls[..., None]
    mom_rhs[:, :, :] += -faces["pstar_n"][..., None] * n_n * Ln[..., None]

    ener_rhs[:, :] += -(faces["pstar_w"] * (faces["u_vec_w"][...,0])) * Lw
    ener_rhs[:, :] += -(faces["pstar_e"] * (faces["u_vec_e"][...,0])) * Le
    ener_rhs[:, :] += -(faces["pstar_s"] * (faces["u_vec_s"][...,1])) * Ls
    ener_rhs[:, :] += -(faces["pstar_n"] * (faces["u_vec_n"][...,1])) * Ln

    return mom_rhs, ener_rhs
