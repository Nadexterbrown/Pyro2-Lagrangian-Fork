import numpy as np

def accumulate_cell_updates(mesh, ufx, pfx, ufy, pfy, gamma):
    """Sum pressure forces/work over faces into per-cell updates.
    ufx,pfx: vertical faces (ny, nx+1) normal speed & star pressure
    ufy,pfy: horizontal faces (ny+1, nx)
    Returns d(rho_u), d(rho_v), d(rho_E) arrays (ny,nx).
    """
    ny, nx = mesh.ny, mesh.nx
    dru = np.zeros((ny,nx))
    drv = np.zeros((ny,nx))
    dE  = np.zeros((ny,nx))
    # vertical faces: normal = +x from left to right, area = face length = dy
    for j in range(ny):
        for i in range(nx+1):
            p = pfx[j,i]; un = ufx[j,i]
            L = mesh.dy[j, min(i, nx-1)] if i<nx else mesh.dy[j, nx-1]
            Fx = - p * L  # force on left cell in +x direction (outward normal)
            W  = - p * un * L
            # left cell (i-1)
            if i>0:
                dru[j,i-1] += Fx
                dE[j,i-1]  += W
            # right cell (i)
            if i<nx:
                dru[j,i]   -= Fx
                dE[j,i]    -= W
    # horizontal faces: normal = +y from bottom to top, area = dx
    for j in range(ny+1):
        for i in range(nx):
            p = pfy[j,i]; un = ufy[j,i]
            L = mesh.dx[min(j, ny-1), i] if j<ny else mesh.dx[ny-1, i]
            Fy = - p * L
            W  = - p * un * L
            if j>0:
                drv[j-1,i] += Fy
                dE[j-1,i]  += W
            if j<ny:
                drv[j,i]   -= Fy
                dE[j,i]    -= W
    return dru, drv, dE
