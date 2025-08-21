import numpy as np
from .riemann import wall_contact_linear

class BoundaryManager:
    def __init__(self, rp):
        self.sides = {
            'xlb': rp.get_param('mesh.xlboundary', 'reflect'),
            'xrb': rp.get_param('mesh.xrboundary', 'outflow'),
            'ylb': rp.get_param('mesh.ylboundary', 'reflect'),
            'yrb': rp.get_param('mesh.yrboundary', 'outflow'),
        }
        self.piston = {
            "type": rp.get_param("piston.kind", "none"),
            "U": float(rp.get_param("piston.U", 0.0)),
            "A": float(rp.get_param("piston.A", 0.0)),
            "f": float(rp.get_param("piston.f", 0.0)),
            "ramp": float(rp.get_param("piston.rampTime", 0.0)),
        }

    def wall_speed(self, t):
        kind = self.piston["type"]
        if kind == "constant":
            return self.piston["U"]
        if kind == "sine":
            return self.piston["A"] * np.sin(2*np.pi*self.piston["f"]*t)
        return 0.0

    def apply_vertical_boundary(self, side, j, state, gamma, t):
        # Return (u_n_star, p_star) for boundary face at column index side
        u = state.u[j, 0 if side=='xlb' else -1]
        p = state.p[j, 0 if side=='xlb' else -1]
        rho = state.rho[j, 0 if side=='xlb' else -1]
        if side=='xlb' and self.sides['xlb'] == 'piston':
            uw = self.wall_speed(t)
            return wall_contact_linear(gamma, rho, u, p, uw)
        # reflecting wall by default
        if self.sides[side] == 'reflect':
            return wall_contact_linear(gamma, rho, u, p, 0.0)
        # outflow: zero-gradient -> take interior as star
        return u, p

    def apply_horizontal_boundary(self, side, i, state, gamma, t):
        v = state.v[0 if side=='ylb' else -1, i]
        p = state.p[0 if side=='ylb' else -1, i]
        rho = state.rho[0 if side=='ylb' else -1, i]
        if self.sides[side] == 'reflect':
            return wall_contact_linear(gamma, rho, v, p, 0.0)
        return v, p
