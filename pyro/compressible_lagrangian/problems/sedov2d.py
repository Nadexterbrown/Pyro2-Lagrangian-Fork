
import numpy as np
from ..state import cons_from_prim

def init_data(ccdata, rp):
    gamma = rp.get_param("eos.gamma", 1.4)
    ny, nx = ccdata._s.mesh.ny, ccdata._s.mesh.nx
    rho = np.ones((ny,nx))
    u = np.zeros((ny,nx))
    v = np.zeros((ny,nx))
    p = 1.0e-6*np.ones((ny,nx))
    j0, i0 = ny//2, nx//2
    p[j0, i0] = 1.0
    mom, Et = cons_from_prim(gamma, rho, u, v, p)
    ccdata.get_var("density")[:] = rho
    ccdata.get_var("x-momentum")[:] = mom[...,0]
    ccdata.get_var("y-momentum")[:] = mom[...,1]
    ccdata.get_var("energy")[:] = Et
