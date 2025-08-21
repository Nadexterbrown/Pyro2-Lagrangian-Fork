
import numpy as np
from ..state import cons_from_prim

def init_data(ccdata, rp):
    gamma = rp.get_param("eos.gamma", 5.0/3.0)
    ny, nx = ccdata._s.mesh.ny, ccdata._s.mesh.nx
    Xc = ccdata._s.mesh.cell_centers()[...,0]
    Yc = ccdata._s.mesh.cell_centers()[...,1]
    xm = 0.5*(ccdata._s.mesh.xmin+ccdata._s.mesh.xmax)
    ym = 0.5*(ccdata._s.mesh.ymin+ccdata._s.mesh.ymax)
    r = np.hypot(Xc-xm, Yc-ym)
    rho = np.ones((ny,nx))
    u = -(Xc-xm) / (r+1e-12)
    v = -(Yc-ym) / (r+1e-12)
    p = 1.0e-6*np.ones_like(rho)
    mom, Et = cons_from_prim(gamma, rho, u, v, p)
    ccdata.get_var("density")[:] = rho
    ccdata.get_var("x-momentum")[:] = mom[...,0]
    ccdata.get_var("y-momentum")[:] = mom[...,1]
    ccdata.get_var("energy")[:] = Et
