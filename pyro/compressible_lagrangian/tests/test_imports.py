def test_import():
    import pyro.compressible_lagrangian_pure as clp
    assert hasattr(clp, "Simulation")
