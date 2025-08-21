import numpy as np
from .eos import GammaLawEOS
from .reconstruction import reconstruct_muscl
from .riemann import riemann as riemann_interface
from .mesh import LagrangianMesh1D
from .state import State1D
from .boundary import PistonBC
class VarView:
    def __init__(self,data,grid): self._data=data; self.g=grid
    def v(self): return self._data
class GridView:
    def __init__(self,xf): self.x=xf; self.ilo=0; self.ihi=xf.size-2
class Simulation:
    def __init__(self,rp):
        def _get(k,default=None):
            try: return rp.get_param(k)
            except Exception: return rp.get(k,default)
        self.gamma=float(_get("eos.gamma",1.4))
        self.eos=GammaLawEOS(self.gamma,p_floor=_get("p_floor",1e-16),rho_floor=_get("rho_floor",1e-16))
        nx=int(_get("mesh.nx",256)); x1=float(_get("mesh.xmax",1.0))
        rho0=float(_get("ic.density",1.0)); u0=float(_get("ic.velocity",0.0)); p0=float(_get("ic.pressure",1.0))
        tau0=1.0/rho0; e0=p0*tau0/(self.gamma-1.0); E0=e0+0.5*u0*u0
        self.mesh=LagrangianMesh1D(0.0,x1,nx,rho0); self.state=State1D(nx,tau0,u0,E0)
        self.t=0.0; self.cfl=float(_get("driver.cfl",0.8))
        self.max_steps=int(_get("driver.max_steps",100000)); self.tmax=float(_get("driver.tmax",0.1))
        self.bc_name_left=_get("mesh.xlboundary","outflow"); self.bc_name_right=_get("mesh.xrboundary","outflow")
        self.piston_fn=_get("piston.bc_fn",lambda t:0.0); self.left_piston=PistonBC(self.eos,self.gamma,self.piston_fn)
        self._limiter={0:"minmod",1:"mc",2:"vanleer"}.get(int(_get("compressible.limiter",2)),"mc")
    def primitives(self): return self.state.primitives(self.eos)
    def compute_dt(self):
        prim=self.primitives(); a=self.eos.sound_speed(prim["tau"],prim["e"])
        dt=self.cfl*self.mesh.dm.min()/(np.max(a)+1e-30)
        u=prim["u"]; u_iface=np.pad(0.5*(u[:-1]+u[1:]),(1,1),mode="edge")
        dt_geom=0.5*self.mesh.dx_min/(np.max(np.abs(u_iface))+1e-30)
        return min(dt,dt_geom)
    def apply_bc(self,prim):
        if self.bc_name_left=="piston":
            tau_g,u_g,p_g=self.left_piston.ghost_left({"tau":prim["tau"][0],"u":prim["u"][0],"p":prim["p"][0]},self.t)
            prim["tau"]=np.concatenate(([tau_g],prim["tau"]))
            prim["u"]=np.concatenate(([u_g],prim["u"]))
            prim["p"]=np.concatenate(([p_g],prim["p"]))
            prim["e"]=np.concatenate(([p_g*tau_g/(self.gamma-1.0)],prim["e"]))
        else:
            for k in ("tau","u","p","e"): prim[k]=np.concatenate(([prim[k][0]],prim[k]))
        for k in ("tau","u","p","e"): prim[k]=np.concatenate((prim[k],[prim[k][-1]]))
        return prim
    def single_step(self):
        prim=self.apply_bc(self.primitives())
        recon=reconstruct_muscl({"tau":prim["tau"],"u":prim["u"],"p":prim["p"]}, limiter=self._limiter)
        tauL,tauR=recon["tau"]; uL,uR=recon["u"]; pL,pR=recon["p"]
        ni=tauL.size; F=np.zeros((ni,3)); u_iface=np.zeros(ni)
        for i in range(ni):
            Floc, ui, _ = riemann_interface(self.eos,{"tau":tauL[i],"u":uL[i],"p":pL[i]},
                                                     {"tau":tauR[i],"u":uR[i],"p":pR[i]})
            F[i,:]=Floc; u_iface[i]=ui
        dt=self.compute_dt(); Fm=np.vstack([F[0,:],F,F[-1,:]])
        dU=-(Fm[1:,:]-Fm[:-1,:])/self.mesh.dm[:,None]
        self.state.tau += dt*dU[:,0]; self.state.u += dt*dU[:,1]; self.state.E += dt*dU[:,2]
        faces=np.zeros(self.mesh.nx+1); faces[1:-1]=u_iface
        faces[0]=float(self.piston_fn(self.t+0.5*dt)) if self.bc_name_left=="piston" else faces[1]
        faces[-1]=faces[-2]; self.mesh.update_faces(faces,dt); self.t+=dt; return dt
    def get_var(self,name):
        if name=="density": arr=(1.0/self.state.tau)[:,None]; return VarView(arr,GridView(self.mesh.xf))
        if name=="pressure": p=self.primitives()["p"][:,None]; return VarView(p,GridView(self.mesh.xf))
        if name=="energy":
            rho=1.0/self.state.tau; arr=(rho*self.state.E)[:,None]; return VarView(arr,GridView(self.mesh.xf))
        if name=="velocity": return [VarView(self.state.u[:,None],GridView(self.mesh.xf))]
        raise KeyError(name)
    def fill_boundary(self): pass
