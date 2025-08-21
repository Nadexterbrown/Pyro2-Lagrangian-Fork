def test_shapes():
    from pyro.compressible_lagrangian_pure.simulation import Simulation
    class RP:
        def __init__(self, d): self.d=d
        def get_param(self, k, default=None): return self.d.get(k, default)
    rp = RP({
        "mesh.nx": 8, "mesh.ny": 4, "mesh.xmax": 1.0, "mesh.ymax": 1.0,
        "eos.gamma": 1.4, "driver.cfl": 0.5,
        "mesh.xlboundary":"reflect", "mesh.xrboundary":"outflow",
        "mesh.ylboundary":"reflect", "mesh.yrboundary":"outflow"
    })
    from pyro.compressible_lagrangian_pure.problems.piston2d import init_data
    sim = Simulation("compressible_lagrangian_pure","piston2d", init_data, rp)
    sim.initialize()
    sim.compute_timestep()
    sim.evolve(sim.dt)
    assert sim.state.rho.shape == (rp.get_param("mesh.ny"), rp.get_param("mesh.nx"))
