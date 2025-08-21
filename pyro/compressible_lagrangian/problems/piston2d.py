import numpy as np
def init_data(my_data, rp):
    gamma = float(rp.get_param("eos.gamma"))
    rho0 = float(rp.get_param("ic.density", 1.0)) if hasattr(rp,'params') else 1.0
    p0   = float(rp.get_param("ic.pressure", 1.0)) if hasattr(rp,'params') else 1.0
    u0   = float(rp.get_param("ic.velocity", 0.0)) if hasattr(rp,'params') else 0.0
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    dens[:,:] = rho0
    xmom[:,:] = rho0 * u0
    ymom[:,:] = 0.0
    ener[:,:] = p0/(gamma-1.0) + 0.5*rho0*(u0*u0)
