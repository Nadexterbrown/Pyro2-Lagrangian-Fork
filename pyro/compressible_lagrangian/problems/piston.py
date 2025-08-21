def piston_bc_fn_factory(vl=1500.0,ramp_time=1e-8):
    def u_w(t): return vl*t/ramp_time if t<ramp_time else vl
    return u_w

def init_data(sim,rp): pass
