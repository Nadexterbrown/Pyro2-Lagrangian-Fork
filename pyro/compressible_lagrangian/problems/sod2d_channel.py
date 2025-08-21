
import numpy as np
from ..state import cons_from_prim

def init_data(ccdata, rp):
    gamma = rp.get_param("eos.gamma", 1.4)
    ny, nx = ccdata._s.mesh.ny, ccdata._s.mesh.nx
    Xc = ccdata._s.mesh.cell_centers()[...,0]
    xmid = 0.5*(ccdata._s.mesh.xmin + ccdata._s.mesh.xmax)
    rhoL, uL, vL, pL = 1.0, 0.0, 0.0, 1.0
    rhoR, uR, vR, pR = 0.125, 0.0, 0.0, 0.1
    rho = np.where(Xc < xmid, rhoL, rhoR)
    u   = np.where(Xc < xmid, uL, uR)
    v   = np.where(Xc < xmid, vL, vR)
    p   = np.where(Xc < xmid, pL, pR)
    mom, Et = cons_from_prim(gamma, rho, u, v, p)
    ccdata.get_var("density")[:] = rho
    ccdata.get_var("x-momentum")[:] = mom[...,0]
    ccdata.get_var("y-momentum")[:] = mom[...,1]
    ccdata.get_var("energy")[:] = Et
