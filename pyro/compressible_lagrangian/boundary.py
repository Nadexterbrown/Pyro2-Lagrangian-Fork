
from __future__ import annotations
import numpy as np

class BoundaryManager:
    def __init__(self, rp):
        self.kind = rp.get_param("piston.kind", "none")
        self.U = float(rp.get_param("piston.U", 0.0))
        self.A = float(rp.get_param("piston.A", 0.0))
        self.f = float(rp.get_param("piston.f", 0.0))
        self.ramp = float(rp.get_param("piston.rampTime", 0.0))
        self.side = rp.get_param("piston.side", "left")  # "left" or "right"

    def piston_speed(self, t):
        if self.kind == "constant":
            vel = self.U
        elif self.kind == "sine":
            vel = self.U + self.A*np.sin(2.0*np.pi*self.f*t)
        else:
            return 0.0
        if t < self.ramp:
            return vel * (t/self.ramp)
        return vel

    def apply(self, mesh, state, t):
        u = state.u
        if self.kind != "none":
            up = self.piston_speed(t)
            if self.side == "left":
                u[:, 0, 0] = up
            else:
                u[:, -1, 0] = up
        # Periodic in y
        u[0, :, 1] = u[1, :, 1]; u[-1, :, 1] = u[-2, :, 1]
        # Outflow at opposite x: zero-gradient
        u[:, -1, 0] = u[:, -2, 0]
