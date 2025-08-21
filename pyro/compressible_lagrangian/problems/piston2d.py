import numpy as np
def init_data(my_data, rp):
    gamma = rp.get_param("eos.gamma")
    rho0 = rp.get_param("ic.density")
    p0   = rp.get_param("ic.pressure")
    u0   = rp.get_param("ic.velocity")
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    dens[:,:] = rho0
    xmom[:,:] = rho0 * u0
    ymom[:,:] = 0.0
    ener[:,:] = p0/(gamma-1.0) + 0.5*rho0*(u0*u0)
