
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class MovingQuadMesh:
    nx: int
    ny: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __init__(self, nx, ny, xmin, xmax, ymin, ymax):
        self.nx, self.ny = nx, ny
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        # Node coordinates on a structured grid (ny+1, nx+1, 2)
        xs = np.linspace(xmin, xmax, nx+1)
        ys = np.linspace(ymin, ymax, ny+1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        self.nodes = np.stack([X, Y], axis=-1)

    # Cell-centered geometry
    @property
    def nc(self):
        return self.ny, self.nx

    def cell_centers(self):
        Xc = 0.25*(self.nodes[:-1, :-1, 0] + self.nodes[1:, :-1, 0] +
                   self.nodes[:-1,  1:, 0] + self.nodes[1:,  1:, 0])
        Yc = 0.25*(self.nodes[:-1, :-1, 1] + self.nodes[1:, :-1, 1] +
                   self.nodes[:-1,  1:, 1] + self.nodes[1:,  1:, 1])
        return np.stack([Xc, Yc], axis=-1)  # (ny, nx, 2)

    def cell_area(self):
        # Bilinear quad area via two triangles
        A = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                n00 = self.nodes[j, i]
                n10 = self.nodes[j, i+1]
                n01 = self.nodes[j+1, i]
                n11 = self.nodes[j+1, i+1]
                # two triangles: (n00,n10,n11) and (n00,n11,n01)
                A[j,i] = 0.5*np.abs(np.cross(n10-n00, n11-n00)) +                          0.5*np.abs(np.cross(n11-n00, n01-n00))
        return A

    def face_geometry(self):
        """Return per-face normals and lengths for west/east/south/north."""
        ny, nx = self.ny, self.nx
        # West/East vertical faces
        Lw = np.linalg.norm(self.nodes[1:, :-1] - self.nodes[:-1, :-1], axis=-1)
        Le = np.linalg.norm(self.nodes[1:, 1:]  - self.nodes[:-1, 1:],  axis=-1)
        # South/North horizontal faces
        Ls = np.linalg.norm(self.nodes[:-1, 1:] - self.nodes[:-1, :-1], axis=-1)
        Ln = np.linalg.norm(self.nodes[1:,  1:] - self.nodes[1:,  :-1], axis=-1)

        # Normals in initial layout (unit)
        nw = np.dstack([-np.ones_like(Lw), np.zeros_like(Lw)]); nw /= np.linalg.norm(nw,axis=-1,keepdims=True)
        ne = np.dstack([ np.ones_like(Le), np.zeros_like(Le)]); ne /= np.linalg.norm(ne,axis=-1,keepdims=True)
        ns = np.dstack([ np.zeros_like(Ls),-np.ones_like(Ls)]); ns /= np.linalg.norm(ns,axis=-1,keepdims=True)
        nn = np.dstack([ np.zeros_like(Ln), np.ones_like(Ln)]); nn /= np.linalg.norm(nn,axis=-1,keepdims=True)

        return (nw, ne, ns, nn), (Lw, Le, Ls, Ln)

    def move_nodes(self, faces, dt):
        """Move nodes by averaging adjacent face velocities (simple, robust)."""
        ny, nx = self.ny, self.nx
        u_node = np.zeros_like(self.nodes)
        w_node = np.zeros(self.nodes.shape[:-1])

        u_w = faces["u_vec_w"]  # (ny, nx, 2)
        u_e = faces["u_vec_e"]
        u_s = faces["u_vec_s"]
        u_n = faces["u_vec_n"]

        # Vertical faces -> add to left/right nodes
        u_node[:-1, :-1] += u_w; w_node[:-1, :-1] += 1.0
        u_node[ 1:, :-1] += u_w; w_node[ 1:, :-1] += 1.0
        u_node[:-1,  1:] += u_e; w_node[:-1,  1:] += 1.0
        u_node[ 1:,  1:] += u_e; w_node[ 1:,  1:] += 1.0

        # Horizontal faces -> add to south/north nodes
        u_node[:-1, :-1] += u_s; w_node[:-1, :-1] += 1.0
        u_node[:-1,  1:] += u_s; w_node[:-1,  1:] += 1.0
        u_node[ 1:, :-1] += u_n; w_node[ 1:, :-1] += 1.0
        u_node[ 1:,  1:] += u_n; w_node[ 1:,  1:] += 1.0

        w_node = np.maximum(w_node, 1e-30)
        u_node /= w_node[..., None]
        self.nodes += dt * u_node

    def inscribed_diameter(self):
        A = self.cell_area()
        return np.sqrt(4.0*A/np.pi)
