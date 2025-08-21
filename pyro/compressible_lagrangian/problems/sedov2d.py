import numpy as np
def init_data(sim):
    mesh = sim.mesh; state = sim.state
    rho0 = 1.0; p0 = 1e-6
    state.m[:,:] = rho0 * mesh.area
    gamma = state.gamma
    state.rho_u[:,:] = 0.0
    state.rho_v[:,:] = 0.0
    # deposit energy at center cell
    E0 = 1.0
    j = mesh.ny//2; i = mesh.nx//2
    state.rho_E[:,:] = p0/(gamma-1.0)
    state.rho_E[j,i] += E0 / (mesh.area[j,i] + 1e-30)
