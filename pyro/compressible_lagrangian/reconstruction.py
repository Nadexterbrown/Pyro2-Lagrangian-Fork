import numpy as np
_PYRO_LIMITERS={}
try:
    from pyro.compressible import reconstruction as _pr
    for nm in ("minmod","mc","vanleer"):
        if hasattr(_pr,nm): _PYRO_LIMITERS[nm]=getattr(_pr,nm)
except Exception: pass

def _minmod(a,b):
    s=np.sign(a)+np.sign(b); return 0.5*s*np.minimum(np.abs(a),np.abs(b))

def _mc(a,b): return _minmod(2*a,_minmod(0.5*(a+b),2*b))

def _vanleer(a,b):
    out=np.zeros_like(a); m=(a*b)>0.0; out[m]=2*a[m]*b[m]/(a[m]+b[m]); return out

LIMITERS={"minmod":_PYRO_LIMITERS.get("minmod",_minmod),
          "mc":_PYRO_LIMITERS.get("mc",_mc),
          "vanleer":_PYRO_LIMITERS.get("vanleer",_vanleer)}

def muscl_reconstruct_1d(q, limiter="mc"):
    lm=LIMITERS.get(limiter, LIMITERS["mc"])
    dqL=q[1:-1]-q[0:-2]; dqR=q[2:]-q[1:-1]; slope=lm(dqL,dqR)
    qL=q[1:-1]-0.5*slope; qR=q[1:-1]+0.5*slope
    return qR[:-1], qL[1:]

def reconstruct_muscl(prim, limiter="mc"):
    return {k: muscl_reconstruct_1d(prim[k], limiter) for k in prim}
