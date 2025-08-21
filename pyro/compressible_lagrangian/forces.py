import numpy as np
def accumulate_cell_updates(mesh, ufx, pfx, ufy, pfy, gamma):
    ny, nx = mesh.ny, mesh.nx
    dru = np.zeros((ny,nx)); drv = np.zeros((ny,nx)); dE  = np.zeros((ny,nx))
    for j in range(ny):
        for i in range(nx+1):
            p = pfx[j,i]; un = ufx[j,i]
            L = mesh.dy[j, min(i, nx-1)] if i<nx else mesh.dy[j, nx-1]
            Fx = - p * L; W  = - p * un * L
            if i>0:  dru[j,i-1] += Fx; dE[j,i-1]  += W
            if i<nx: dru[j,i]   -= Fx; dE[j,i]    -= W
    for j in range(ny+1):
        for i in range(nx):
            p = pfy[j,i]; un = ufy[j,i]
            L = mesh.dx[min(j, ny-1), i] if j<ny else mesh.dx[ny-1, i]
            Fy = - p * L; W  = - p * un * L
            if j>0:  drv[j-1,i] += Fy; dE[j-1,i]  += W
            if j<ny: drv[j,i]   -= Fy; dE[j,i]    -= W
    return dru, drv, dE
