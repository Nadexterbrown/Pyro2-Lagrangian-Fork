from pyro.compressible_lagrangian.simulation import Simulation

def test_smoke():
    rp={"eos.gamma":1.4,"mesh.nx":64,"mesh.xmax":1.0,"ic.density":1.0,"ic.velocity":0.0,"ic.pressure":1.0,
        "driver.cfl":0.6,"driver.tmax":1e-3,"mesh.xlboundary":"outflow","mesh.xrboundary":"outflow"}
    sim=Simulation(rp)
    for _ in range(5): sim.single_step()
    assert sim.t>0.0
