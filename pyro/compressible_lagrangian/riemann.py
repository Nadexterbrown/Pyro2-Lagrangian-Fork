
from __future__ import annotations
import numpy as np

def hllc_1d(gamma, rL,uL,pL, rR,uR,pR):
    aL = np.sqrt(gamma*np.maximum(pL,0.0)/np.maximum(rL,1e-30))
    aR = np.sqrt(gamma*np.maximum(pR,0.0)/np.maximum(rR,1e-30))
    SL = min(uL - aL, uR - aR)
    SR = max(uL + aL, uR + aR)
    # PVRS estimate for p*
    pPV = 0.5*(pL+pR) - 0.5*(uR-uL)*0.5*(rL+rR)*0.5*(aL+aR)
    pStar = max(0.0, pPV)
    denom = (rL*(SL-uL) - rR*(SR-uR))
    if abs(denom) < 1e-30:
        Sstar = 0.5*(uL+uR)
    else:
        Sstar = (pR - pL + rL*uL*(SL-uL) - rR*uR*(SR-uR)) / denom
    return Sstar, pStar

def face_states_and_star(mesh, gamma, prim, grad=None):
    """Compute star normal velocity and star pressure on each face."""
    rho, u, v, p = prim
    ny, nx = rho.shape
    u_vec_w = np.zeros((ny, nx, 2))
    u_vec_e = np.zeros((ny, nx, 2))
    u_vec_s = np.zeros((ny, nx, 2))
    u_vec_n = np.zeros((ny, nx, 2))
    pstar_w = np.zeros((ny, nx))
    pstar_e = np.zeros((ny, nx))
    pstar_s = np.zeros((ny, nx))
    pstar_n = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            # WEST
            if i > 0:
                rL,uL,vL,pL = rho[j,i-1], u[j,i-1], v[j,i-1], p[j,i-1]
                rR,uR,vR,pR = rho[j,i],   u[j,i],   v[j,i],   p[j,i]
                Sstar, pstar = hllc_1d(gamma, rL,uL,pL, rR,uR,pR)
                u_vec_w[j,i] = np.array([Sstar, 0.0]); pstar_w[j,i] = pstar
            else:
                u_vec_w[j,i] = np.array([u[j,i], 0.0]); pstar_w[j,i] = p[j,i]
            # EAST
            if i < nx-1:
                rL,uL,vL,pL = rho[j,i],   u[j,i],   v[j,i],   p[j,i]
                rR,uR,vR,pR = rho[j,i+1], u[j,i+1], v[j,i+1], p[j,i+1]
                Sstar, pstar = hllc_1d(gamma, rL,uL,pL, rR,uR,pR)
                u_vec_e[j,i] = np.array([Sstar, 0.0]); pstar_e[j,i] = pstar
            else:
                u_vec_e[j,i] = np.array([u[j,i], 0.0]); pstar_e[j,i] = p[j,i]
            # SOUTH (normal along y)
            if j > 0:
                rL,uL,vL,pL = rho[j-1,i], v[j-1,i], u[j-1,i], p[j-1,i]
                rR,uR,vR,pR = rho[j,i],   v[j,i],   u[j,i],   p[j,i]
                Sstar, pstar = hllc_1d(gamma, rL,uL,pL, rR,uR,pR)
                u_vec_s[j,i] = np.array([0.0, Sstar]); pstar_s[j,i] = pstar
            else:
                u_vec_s[j,i] = np.array([0.0, v[j,i]]); pstar_s[j,i] = p[j,i]
            # NORTH
            if j < ny-1:
                rL,uL,vL,pL = rho[j,i],   v[j,i],   u[j,i],   p[j,i]
                rR,uR,vR,pR = rho[j+1,i], v[j+1,i], u[j+1,i], p[j+1,i]
                Sstar, pstar = hllc_1d(gamma, rL,uL,pL, rR,uR,pR)
                u_vec_n[j,i] = np.array([0.0, Sstar]); pstar_n[j,i] = pstar
            else:
                u_vec_n[j,i] = np.array([0.0, v[j,i]]); pstar_n[j,i] = p[j,i]
    return {
        "u_vec_w": u_vec_w, "u_vec_e": u_vec_e, "u_vec_s": u_vec_s, "u_vec_n": u_vec_n,
        "pstar_w": pstar_w, "pstar_e": pstar_e, "pstar_s": pstar_s, "pstar_n": pstar_n,
    }
