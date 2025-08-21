import numpy as np
class EquationOfState:
    def pressure(self,tau,e): raise NotImplementedError
    def sound_speed(self,tau,e): raise NotImplementedError
    def temperature(self,tau,e,Y=None): return np.nan*np.ones_like(tau)
class GammaLawEOS(EquationOfState):
    def __init__(self,gamma:float=1.4,p_floor:float=1e-16,rho_floor:float=1e-16):
        self.gamma=float(gamma); self.p_floor=float(p_floor); self.rho_floor=float(rho_floor)
    def pressure(self,tau,e):
        rho=np.maximum(1.0/np.maximum(tau,1.0/self.rho_floor),self.rho_floor)
        p=(self.gamma-1.0)*rho*np.maximum(e,0.0); return np.maximum(p,self.p_floor)
    def sound_speed(self,tau,e):
        p=self.pressure(tau,e); rho=np.maximum(1.0/np.maximum(tau,1.0/self.rho_floor),self.rho_floor)
        return np.sqrt(np.maximum(self.gamma*p/rho,0.0))
