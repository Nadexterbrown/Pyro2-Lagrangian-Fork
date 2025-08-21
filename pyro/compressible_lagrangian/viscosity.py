import numpy as np
def vnr_q_vertical(mesh, state, coeff):
    ny, nx = mesh.ny, mesh.nx
    q = np.zeros((ny, nx+1))
    if coeff <= 0.0: return q
    for j in range(ny):
        for i in range(1, nx):
            dudx = (state.u[j,i] - state.u[j,i-1]) / (mesh.dx[j,i-1] + 1e-30)
            if dudx < 0.0:
                q[j,i] = coeff * state.rho[j,i-1] * (mesh.dx[j,i-1]**2) * (dudx**2)
    return q
def vnr_q_horizontal(mesh, state, coeff):
    ny, nx = mesh.ny, mesh.nx
    q = np.zeros((ny+1, nx))
    if coeff <= 0.0: return q
    for j in range(1, ny):
        for i in range(nx):
            dvdy = (state.v[j,i] - state.v[j-1,i]) / (mesh.dy[j-1,i] + 1e-30)
            if dvdy < 0.0:
                q[j,i] = coeff * state.rho[j-1,i] * (mesh.dy[j-1,i]**2) * (dvdy**2)
    return q
