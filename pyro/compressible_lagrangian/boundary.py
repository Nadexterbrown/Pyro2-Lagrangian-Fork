import numpy as np
from .riemann import wall_contact_linear
from .util import rp_get

class BoundaryManager:
    def __init__(self, rp):
        self.sides = {
            'xlb': rp_get(rp, 'mesh.xlboundary', 'reflect'),
            'xrb': rp_get(rp, 'mesh.xrboundary', 'outflow'),
            'ylb': rp_get(rp, 'mesh.ylboundary', 'reflect'),
            'yrb': rp_get(rp, 'mesh.yrboundary', 'outflow'),
        }
        self.piston = {
            "type": rp_get(rp, "piston.kind", "none"),
            "U": float(rp_get(rp, "piston.U", 0.0)),
            "A": float(rp_get(rp, "piston.A", 0.0)),
            "f": float(rp_get(rp, "piston.f", 0.0)),
            "ramp": float(rp_get(rp, "piston.rampTime", 0.0)),
        }

    def wall_speed(self, t):
        kind = self.piston["type"]
        if kind == "constant":
            return self.piston["U"]
        if kind == "sine":
            import numpy as np
            return self.piston["A"] * np.sin(2*np.pi*self.piston["f"]*t)
        return 0.0

    def apply_vertical_boundary(self, side, j, state, gamma, t):
        u = state.u[j, 0 if side=='xlb' else -1]
        p = state.p[j, 0 if side=='xlb' else -1]
        rho = state.rho[j, 0 if side=='xlb' else -1]
        if side=='xlb' and self.sides['xlb'] == 'piston':
            uw = self.wall_speed(t)
            return wall_contact_linear(gamma, rho, u, p, uw)
        if self.sides[side] == 'reflect':
            return wall_contact_linear(gamma, rho, u, p, 0.0)
        return u, p

    def apply_horizontal_boundary(self, side, i, state, gamma, t):
        v = state.v[0 if side=='ylb' else -1, i]
        p = state.p[0 if side=='ylb' else -1, i]
        rho = state.rho[0 if side=='ylb' else -1, i]
        if self.sides[side] == 'reflect':
            return wall_contact_linear(gamma, rho, v, p, 0.0)
        return v, p
