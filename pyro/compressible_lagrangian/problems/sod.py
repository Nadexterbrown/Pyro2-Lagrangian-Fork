def init_data(sim,rp):
    x=sim.mesh.xc; x0=0.5*sim.mesh.xf[-1]
    rhoL,uL,pL=1.0,0.0,1.0; rhoR,uR,pR=0.125,0.0,0.1
    left=x<x0
    sim.state.tau[left]=1.0/rhoL; sim.state.u[left]=uL
    eL=pL*(1.0/rhoL)/(sim.gamma-1.0); sim.state.E[left]=eL+0.5*uL*uL
    sim.state.tau[~left]=1.0/rhoR; sim.state.u[~left]=uR
    eR=pR*(1.0/rhoR)/(sim.gamma-1.0); sim.state.E[~left]=eR+0.5*uR*uR
