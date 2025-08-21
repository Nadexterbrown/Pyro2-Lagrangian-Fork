import numpy as np
def init_data(sim):
    mesh = sim.mesh; state = sim.state
    gamma = state.gamma
    # 2D Sod along x, extruded in y
    xc_mid = mesh.x_line[mesh.nx//2]
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            if mesh.Xc[j,i] < xc_mid:
                rho = 1.0; p = 1.0; u=0.0; v=0.0
            else:
                rho = 0.125; p = 0.1; u=0.0; v=0.0
            state.m[j,i] = rho * mesh.area[j,i]
            state.rho_u[j,i] = rho*u
            state.rho_v[j,i] = rho*v
            state.rho_E[j,i] = p/(gamma-1.0) + 0.5*rho*(u*u+v*v)
