import numpy as np
class LagrangianMesh1D:
    def __init__(self,x0:float,x1:float,nx:int,rho0:float):
        self.nx=int(nx); self.xf=np.linspace(x0,x1,nx+1); self.xc=0.5*(self.xf[:-1]+self.xf[1:])
        L=(x1-x0); mass_per_area=rho0*L; self.dm=np.full(nx,mass_per_area/nx)
    def update_faces(self,u_iface,dt):
        self.xf += dt*u_iface; self.xc=0.5*(self.xf[:-1]+self.xf[1:])
    @property
    def dx_min(self): return np.min(self.xf[1:]-self.xf[:-1])
