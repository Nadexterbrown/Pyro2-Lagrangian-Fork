from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .util import rp_get

@dataclass
class GridView:
    ilo: int; ihi: int; jlo: int; jhi: int; ng: int
    x: np.ndarray
    y: np.ndarray

class MovingQuadMesh:
    """Structured, logically-rectangular moving quad mesh (axis-aligned deformation)."""
    def __init__(self, rp):
        self.nx = int(rp_get(rp, "mesh.nx"))
        self.ny = int(rp_get(rp, "mesh.ny"))
        self.xmin = float(rp_get(rp, "mesh.xmin", 0.0))
        self.xmax = float(rp_get(rp, "mesh.xmax"))
        self.ymin = float(rp_get(rp, "mesh.ymin", 0.0))
        self.ymax = float(rp_get(rp, "mesh.ymax"))
        self.ng = int(rp_get(rp, "mesh.ng", 1))

        xs = np.linspace(self.xmin, self.xmax, self.nx+1)
        ys = np.linspace(self.ymin, self.ymax, self.ny+1)
        self.Xn, self.Yn = np.meshgrid(xs, ys, indexing="xy")
        self.update_geometry()

    def update_geometry(self):
        self.Xc = 0.5*(self.Xn[:-1,:-1] + self.Xn[1:,1:])
        self.Yc = 0.5*(self.Yn[:-1,:-1] + self.Yn[1:,1:])
        self.dx = np.abs(self.Xn[1:,:-1] - self.Xn[:-1,:-1])
        self.dy = np.abs(self.Yn[:-1,1:] - self.Yn[:-1,:-1])
        self.area = self.dx * self.dy
        self.x_line = self.Xc.mean(axis=0)
        self.y_line = self.Yc.mean(axis=1)

    def pyro_grid_view(self) -> GridView:
        ilo = self.ng
        ihi = self.ng + self.nx - 1
        jlo = self.ng
        jhi = self.ng + self.ny - 1
        return GridView(ilo=ilo, ihi=ihi, jlo=jlo, jhi=jhi, ng=self.ng,
                        x=self.x_line.copy(), y=self.y_line.copy())

    def assemble_nodal_velocity(self, ufx, ufy):
        ny, nx = self.ny, self.nx
        Un = np.zeros((ny+1, nx+1))
        Vn = np.zeros((ny+1, nx+1))
        w = np.zeros((ny+1, nx+1))
        for j in range(ny):
            for i in range(1, nx):
                u = ufx[j, i-1]
                Un[j, i] += u; Un[j+1, i] += u
                w[j, i] += 1.0; w[j+1, i] += 1.0
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
        import numpy as np
        a = np.sqrt(np.maximum(state.gamma * state.p / np.maximum(state.rho, 1e-30), 0.0))
        ell = np.minimum(self.dx, self.dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            dt = cfl * np.nanmin(ell / np.maximum(a, 1e-14))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-12
        return float(dt)
