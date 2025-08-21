from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class GridView:
    # minimal Pyro-like grid view (cell-centered coordinates & index bounds)
    ilo: int; ihi: int; jlo: int; jhi: int; ng: int
    x: np.ndarray  # (nx,) current cell-center x
    y: np.ndarray  # (ny,) current cell-center y

class MovingQuadMesh:
    """Structured, logically-rectangular moving quad mesh (axis-aligned deformation).
    This scaffold supports a purely Lagrangian update while keeping the geometry simple.
    """
    def __init__(self, rp):
        self.nx = int(rp.get_param("mesh.nx"))
        self.ny = int(rp.get_param("mesh.ny"))
        self.xmin = float(rp.get_param("mesh.xmin", 0.0))
        self.xmax = float(rp.get_param("mesh.xmax"))
        self.ymin = float(rp.get_param("mesh.ymin", 0.0))
        self.ymax = float(rp.get_param("mesh.ymax"))
        self.ng = int(rp.get_param("mesh.ng", 1))

        # Node positions (nx+1, ny+1)
        xs = np.linspace(self.xmin, self.xmax, self.nx+1)
        ys = np.linspace(self.ymin, self.ymax, self.ny+1)
        self.Xn, self.Yn = np.meshgrid(xs, ys, indexing="xy")  # node coords

        # Derived cell geometry
        self.update_geometry()

    def update_geometry(self):
        # cell-centered coords and areas (axis-aligned quads)
        xc = 0.5*(self.Xn[:-1,:-1] + self.Xn[1:,1:])  # crude but ok for axis aligned
        yc = 0.5*(self.Yn[:-1,:-1] + self.Yn[1:,1:])
        self.Xc = xc.copy()
        self.Yc = yc.copy()
        # cell widths
        dx = self.Xn[1:,:-1] - self.Xn[:-1,:-1]  # (ny, nx)
        dy = self.Yn[:-1,1:] - self.Yn[:-1,:-1]  # (ny, nx)
        # For axis-aligned grids, we can take average |dx| and |dy| per cell
        self.dx = np.abs(dx)
        self.dy = np.abs(dy)
        self.area = self.dx * self.dy

        # Precompute 1D cell-centered coordinates for GridView
        # Use simple averages along y/x respectively (sufficient for diagnostics)
        self.x_line = self.Xc.mean(axis=0)  # (nx,)
        self.y_line = self.Yc.mean(axis=1)  # (ny,)

    def pyro_grid_view(self) -> GridView:
        ilo = self.ng
        ihi = self.ng + self.nx - 1
        jlo = self.ng
        jhi = self.ng + self.ny - 1
        return GridView(ilo=ilo, ihi=ihi, jlo=jlo, jhi=jhi, ng=self.ng,
                        x=self.x_line.copy(), y=self.y_line.copy())

    def assemble_nodal_velocity(self, ufx, ufy):
        """Area-weighted nodal velocity from adjacent face star velocities.
        ufx: face-normal speeds on vertical faces at i=1..nx-1 (ny rows)
        ufy: face-normal speeds on horizontal faces at j=1..ny-1 (nx cols)
        Returns arrays Un, Vn of shape (ny+1, nx+1).
        """
        ny, nx = self.ny, self.nx
        Un = np.zeros((ny+1, nx+1))
        Vn = np.zeros((ny+1, nx+1))
        w = np.zeros((ny+1, nx+1))
        # vertical faces contribute to x-velocity at neighboring nodes
        for j in range(ny):
            for i in range(1, nx):
                u = ufx[j, i-1]
                Un[j, i] += u; Un[j+1, i] += u
                w[j, i] += 1.0; w[j+1, i] += 1.0
        # horizontal faces contribute to y-velocity
        for j in range(1, ny):
            for i in range(nx):
                v = ufy[j-1, i]
                Vn[j, i] += v; Vn[j, i+1] += v
                w[j, i] += 1.0; w[j, i+1] += 1.0
        w[w==0.0] = 1.0
        Un /= w; Vn /= w
        return Un, Vn

    def move_nodes(self, Un, Vn, dt):
        self.Xn += dt * Un
        self.Yn += dt * Vn
        self.update_geometry()

    def cfl_timestep(self, state, cfl: float):
        a = np.sqrt(np.maximum(state.gamma * state.p / np.maximum(state.rho, 1e-30), 0.0))
        # characteristic length: min(dx,dy)
        ell = np.minimum(self.dx, self.dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            dt = cfl * np.nanmin(ell / np.maximum(a, 1e-14))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-12
        return float(dt)
