import numpy as np
class State1D:
    def __init__(self,nx,tau0,u0,E0):
        self.nx=nx; self.tau=np.full(nx,tau0); self.u=np.full(nx,u0); self.E=np.full(nx,E0)
    def primitives(self,eos):
        e=self.E-0.5*self.u*self.u; p=eos.pressure(self.tau,e)
        return {"tau":self.tau.copy(),"u":self.u.copy(),"p":p.copy(),"e":e.copy()}
