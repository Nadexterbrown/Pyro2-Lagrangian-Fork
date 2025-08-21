
import numpy as np
from ..simulation import Simulation

class RP:
    def __init__(self, d):
        self.d=d
    def get_param(self,k,default=None):
        return self.d.get(k,default)

def test_construction():
    rp = RP({
        "mesh.nx": 16, "mesh.ny": 8,
        "mesh.xmax": 1.0, "mesh.ymax": 0.1,
        "eos.gamma": 1.4,
    })
    def dummy(cc, rp): pass
    sim = Simulation("compressible_lagrangian_pure", "dummy", dummy, rp)
    sim.initialize()
    assert sim.mesh.nx == 16 and sim.mesh.ny == 8
