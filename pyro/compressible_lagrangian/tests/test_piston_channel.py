
import numpy as np
from ..simulation import Simulation

class RP:
    def __init__(self, d):
        self.d=d
    def get_param(self,k,default=None):
        return self.d.get(k,default)

def init_piston(cc, rp):
    ny, nx = cc._s.mesh.ny, cc._s.mesh.nx
    cc.get_var("density")[:] = 1.0
    cc.get_var("x-momentum")[:] = 0.0
    cc.get_var("y-momentum")[:] = 0.0
    cc.get_var("energy")[:] = 1.0/(1.4-1.0)

def test_step():
    rp = RP({
        "mesh.nx": 8, "mesh.ny": 4,
        "mesh.xmax": 1.0, "mesh.ymax": 0.1,
        "eos.gamma": 1.4,
        "driver.cfl": 0.5, "driver.tmax": 1e-3,
        "piston.kind": "constant", "piston.U": 0.1, "piston.side": "left"
    })
    sim = Simulation("compressible_lagrangian_pure", "piston", init_piston, rp)
    sim.initialize()
    dt = sim.compute_timestep()
    sim.evolve(dt)
    assert np.isfinite(sim.t)
