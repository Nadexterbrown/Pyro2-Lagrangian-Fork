
from __future__ import annotations
import numpy as np

def minmod(a, b):
    s = np.sign(a) + np.sign(b)
    return 0.5*s*np.minimum(np.abs(a), np.abs(b))

def muscl_reconstruct(mesh, prim):
    """Skeleton MUSCL; returns cell-centered primitives (first order)."""
    rho, u, v, p = prim
    return {"rho": rho, "u": u, "v": v, "p": p}
