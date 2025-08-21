import numpy as np

def _sound_speed(eos,tau,e): return eos.sound_speed(tau,e)

def _acoustic_interface(eos,tauL,uL,pL,tauR,uR,pR):
    g=getattr(eos,"gamma",1.4); eL=pL*tauL/(g-1.0); eR=pR*tauR/(g-1.0)
    aL=_sound_speed(eos,tauL,eL); aR=_sound_speed(eos,tauR,eR)
    ZL, ZR=aL, aR
    ustar=(ZL*uL+ZR*uR+pL-pR)/(ZL+ZR+1e-30)
    pstar=(ZR*pL+ZL*pR+ZL*ZR*(uL-uR))/(ZL+ZR+1e-30)
    return ustar,pstar,aL,aR

def _flux_lag(tau,u,p): return np.array([-u,p,p*u])

def riemann(eos,qL,qR,aux=None):
    tauL,uL,pL = qL["tau"], qL["u"], qL["p"]
    tauR,uR,pR = qR["tau"], qR["u"], qR["p"]
    ustar,pstar,aL,aR=_acoustic_interface(eos,tauL,uL,pL,tauR,uR,pR)
    SL,SR=-aL,+aR
    FL=_flux_lag(tauL,uL,pL); FR=_flux_lag(tauR,uR,pR)
    g=getattr(eos,"gamma",1.4)
    UL=np.array([tauL,uL,pL*tauL/(g-1.0)+0.5*uL*uL])
    UR=np.array([tauR,uR,pR*tauR/(g-1.0)+0.5*uR*uR])
    if SL>=0.0: return FL,uL,(tauL,uL,pL)
    if SR<=0.0: return FR,uR,(tauR,uR,pR)
    FHLL=(SR*FL - SL*FR + SL*SR*(UR-UL))/(SR-SL+1e-30)
    return FHLL, ustar, (np.nan, ustar, pstar)
