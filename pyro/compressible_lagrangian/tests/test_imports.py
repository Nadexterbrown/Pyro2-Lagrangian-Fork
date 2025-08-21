def test_import():
    import pyro.compressible_lagrangian as cl
    assert hasattr(cl, "Simulation")
