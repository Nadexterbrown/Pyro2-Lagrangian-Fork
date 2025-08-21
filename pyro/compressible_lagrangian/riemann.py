import numpy as np
def pvrs_contact(gamma, rhoL,uL,pL, rhoR,uR,pR):
    aL = np.sqrt(np.maximum(gamma * pL / np.maximum(rhoL,1e-30), 0.0))
    aR = np.sqrt(np.maximum(gamma * pR / np.maximum(rhoR,1e-30), 0.0))
    aBar = 0.5*(aL + aR)
    rhoBar = 0.5*(rhoL + rhoR)
    pStar = max(1e-30, 0.5*(pL+pR) - 0.5*(uR-uL)*rhoBar*aBar)
    uStar = 0.5*(uL+uR) + (pL - pR)/(rhoBar*aBar + 1e-30)
    return uStar, pStar
def wall_contact_linear(gamma, rho, u, p, u_wall):
    a = np.sqrt(np.maximum(gamma * p / np.maximum(rho,1e-30), 0.0))
    pStar = max(1e-30, p + rho*a*(u - u_wall))
    uStar = u_wall
    return uStar, pStar
