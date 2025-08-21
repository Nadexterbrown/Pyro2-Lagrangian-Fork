
"""
Use Pyro's standard Sod initialization to keep parity with Eulerian solver.
If not available, provide a minimal local initializer.
"""
def init_data(sim, rp):
    try:
        from pyro.compressible.problems.sod import init_data as euler_sod_init
        return euler_sod_init(sim, rp)
    except Exception:
        # fallback: simple 1-D Sod along x extruded in y
        import numpy as np
        x = sim.cc_data.grid.x[:, None]
        xc = 0.5*(sim.cc_data.grid.x.min() + sim.cc_data.grid.x.max())
        left = (x < xc)
        rhoL,uL,vL,pL = 1.0,0.0,0.0,1.0
        rhoR,uR,vR,pR = 0.125,0.0,0.0,0.1

        dens = sim.cc_data.get_var("density")
        xmom = sim.cc_data.get_var("x-momentum")
        ymom = sim.cc_data.get_var("y-momentum")
        ener = sim.cc_data.get_var("energy")

        dens[left] = rhoL; xmom[left] = rhoL*uL; ymom[left]=rhoL*vL
        eL = pL/(sim.rp.get_param("eos.gamma", 1.4)-1.0) + 0.5*rhoL*(uL*uL+vL*vL)
        ener[left] = eL

        dens[~left] = rhoR; xmom[~left] = rhoR*uR; ymom[~left]=rhoR*vR
        eR = pR/(sim.rp.get_param("eos.gamma", 1.4)-1.0) + 0.5*rhoR*(uR*uR+vR*vR)
        ener[~left] = eR
