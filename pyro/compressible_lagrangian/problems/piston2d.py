import numpy as np
def init_data(sim):
    mesh = sim.mesh; state = sim.state
    rho0 = 1.0; p0 = 1.0; u0 = 0.0; v0 = 0.0
    state.m[:,:] = rho0 * mesh.area
    state.rho_u[:,:] = rho0*u0
    state.rho_v[:,:] = rho0*v0
    state.rho_E[:,:] = p0/(state.gamma-1.0) + 0.5*rho0*(u0*u0+v0*v0)
