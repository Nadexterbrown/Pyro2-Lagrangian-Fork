import numpy as np
def init_data(sim):
    mesh = sim.mesh; state = sim.state
    rho0 = 1.0; p0 = 1e-6; vr = -1.0  # radial implosion magnitude
    # initialize uniform mass and small pressure
    state.m[:,:] = rho0 * mesh.area
    # radial inward velocity approximate on cartesian grid
    Xc, Yc = mesh.Xc, mesh.Yc
    r = np.sqrt((Xc - Xc.mean())**2 + (Yc - Yc.mean())**2) + 1e-12
    state.rho_u[:,:] = state.m * (vr*(Xc - Xc.mean())/r) / mesh.area
    state.rho_v[:,:] = state.m * (vr*(Yc - Yc.mean())/r) / mesh.area
    gamma = state.gamma
    eint = p0/(gamma-1.0)
    state.rho_E[:,:] = eint + 0.5*state.m/mesh.area*( (state.rho_u/state.m)**2 + (state.rho_v/state.m)**2 )
