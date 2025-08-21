
import numpy as np

__all__ = [
    "consFlux",
    "estimate_wave_speed",
    "riemann_cgf",
    "riemann_flux",
    "riemann_hllc",
    "riemann_hllc_lowspeed",
    "riemann_prim",
]

def _split_normal_tangential(idir, u, v):
    if idir == 1:   # x-normal
        return u, v
    elif idir == 2: # y-normal
        return v, u
    else:
        raise ValueError("idir must be 1 (x-normal) or 2 (y-normal)")

def _merge_normal_tangential(idir, un, ut):
    return (un, ut) if idir == 1 else (ut, un)

def _to_conserved(gamma, rho, u, v, p):
    E = p/(gamma-1.0)/rho + 0.5*(u*u + v*v)
    return rho, rho*u, rho*v, rho*E

def _euler_flux_normal(gamma, rho, un, ut, p):
    E = p/(gamma-1.0)/rho + 0.5*(un*un + ut*ut)
    F_rho = rho*un
    F_mn  = rho*un*un + p
    F_mt  = rho*un*ut
    F_E   = (rho*E + p)*un
    return F_rho, F_mn, F_mt, F_E

def consFlux(idir, gamma, q):
    rho, u, v, p = q
    un, ut = _split_normal_tangential(idir, u, v)
    return _euler_flux_normal(gamma, rho, un, ut, p)

def estimate_wave_speed(gamma, rhoL, uL, pL, rhoR, uR, pR):
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    return SL, SR

def _star_state(gamma, rho, un, ut, p, S, SM):
    E = p/(gamma-1.0)/rho + 0.5*(un*un + ut*ut)
    rhoE = rho*E
    denom = S - SM
    denom = np.where(np.abs(denom) > 1e-14, denom, np.sign(denom)*1e-14)
    rho_star = rho*(S - un)/denom
    mn_star  = rho_star * SM
    mt_star  = rho_star * ut
    p_star   = p + rho*(S - un)*(SM - un)
    E_star   = ((S - un)*rhoE - p*un + p_star*SM)/denom
    return rho_star, mn_star, mt_star, E_star

def _hllc_ale(idir, gamma, qL, qR):
    rhoL, uL, vL, pL = qL
    rhoR, uR, vR, pR = qR

    uLn, uLt = _split_normal_tangential(idir, uL, vL)
    uRn, uRt = _split_normal_tangential(idir, uR, vR)

    FL = np.stack(_euler_flux_normal(gamma, rhoL, uLn, uLt, pL), axis=-1)
    FR = np.stack(_euler_flux_normal(gamma, rhoR, uRn, uRt, pR), axis=-1)

    SL, SR = estimate_wave_speed(gamma, rhoL, uLn, pL, rhoR, uRn, pR)

    num   = pR - pL + rhoL*uLn*(SL - uLn) - rhoR*uRn*(SR - uRn)
    denom = rhoL*(SL - uLn) - rhoR*(SR - uRn)
    denom = np.where(np.abs(denom) > 1e-14, denom, np.sign(denom)*1e-14)
    SM    = num/denom  # contact speed → choose as mesh normal speed

    UL = np.stack(_to_conserved(gamma, rhoL, uL, vL, pL), axis=-1)
    UR = np.stack(_to_conserved(gamma, rhoR, uR, vR, pR), axis=-1)

    rhoL_star, mnL_star, mtL_star, EL_star = _star_state(gamma, rhoL, uLn, uLt, pL, SL, SM)
    rhoR_star, mnR_star, mtR_star, ER_star = _star_state(gamma, rhoR, uRn, uRt, pR, SR, SM)

    uL_star, vL_star = _merge_normal_tangential(idir,
                        mnL_star/np.maximum(rhoL_star,1e-300),
                        mtL_star/np.maximum(rhoL_star,1e-300))
    uR_star, vR_star = _merge_normal_tangential(idir,
                        mnR_star/np.maximum(rhoR_star,1e-300),
                        mtR_star/np.maximum(rhoR_star,1e-300))

    UL_star = np.stack([rhoL_star, rhoL_star*uL_star, rhoL_star*vL_star, rhoL_star*EL_star], axis=-1)
    UR_star = np.stack([rhoR_star, rhoR_star*uR_star, rhoR_star*vR_star, rhoR_star*ER_star], axis=-1)

    # Standard HLLC branching at xi=0 in Eulerian frame
    mask_LL = (0.0 <= SL)
    mask_LS = (SL <= 0.0) & (0.0 <= SM)
    mask_SR = (SM <= 0.0) & (0.0 <= SR)
    mask_RR = (SR <= 0.0)

    F = np.zeros_like(FL)
    U_face = np.zeros_like(UL)

    if np.any(mask_LL):
        F = np.where(mask_LL[..., None], FL, F)
        U_face = np.where(mask_LL[..., None], UL, U_face)

    if np.any(mask_LS):
        F_LS = FL + SL[..., None]*(UL_star - UL)
        m = mask_LS[..., None]
        F = np.where(m, F_LS, F)
        U_face = np.where(m, UL_star, U_face)

    if np.any(mask_SR):
        F_SR = FR + SR[..., None]*(UR_star - UR)
        m = mask_SR[..., None]
        F = np.where(m, F_SR, F)
        U_face = np.where(m, UR_star, U_face)

    if np.any(mask_RR):
        m = mask_RR[..., None]
        F = np.where(m, FR, F)
        U_face = np.where(m, UR, U_face)

    # ALE (truly Lagrangian): subtract mesh advection with w_n = S_M
    w_n = SM
    Smax = np.maximum(np.abs(SL - w_n), np.abs(SR - w_n))
    F_ALE = F - w_n[..., None] * U_face
    return F_ALE, Smax

# ---- Pyro2 public API (names/signatures) ----

def riemann_hllc(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r):
    return _hllc_ale(idir, float(gamma), q_l, q_r)

def riemann_hllc_lowspeed(*args, **kwargs):
    return riemann_hllc(*args, **kwargs)

def riemann_cgf(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r):
    # For now, alias to HLLC but keep the API so Pyro config "riemann=CGF" still works.
    return riemann_hllc(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r)

def riemann_prim(idir, ng, irho, iu, iv, ip, ix, nspec, gamma, q_l, q_r):
    # Legacy wrapper used by some paths
    return riemann_hllc(idir, ng, irho, iu, iv, ip, ix, gamma, q_l, q_r)

def riemann_flux(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r, riemann="HLLC"):
    rr = (riemann or "HLLC").upper()
    if rr in ("HLLC","HLLC_LOWSPEED","LOW","HLLC_LOW"):
        return riemann_hllc(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r)
    elif rr in ("CGF","TWO-SHOCK","TWOSHOCK"):
        return riemann_cgf(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r)
    else:
        return riemann_hllc(idir, ng, irho, ixmom, iymom, iener, gamma, q_l, q_r)

def consFlux(idir, gamma, q):
    return _euler_flux_normal(float(gamma), q[0], *_split_normal_tangential(idir, q[1], q[2]), q[3])
