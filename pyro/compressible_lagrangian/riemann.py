
from pyro.compressible import riemann as euler_riemann

def riemann_flux(dim, U_L, U_R,
                 cc_data, rp, ivars,
                 solid_L=False, solid_R=False, tc=None,
                 return_cons=False):
    try:
        mode = rp.get_param("compressible_lagrangian.mode")
    except Exception:
        mode = "passthrough"

    if mode == "ale":
        # TODO: implement ALE/Lagrangian flux. For now, fall back for stability.
        pass

    return euler_riemann.riemann_flux(dim, U_L, U_R,
                                      cc_data, rp, ivars,
                                      solid_L, solid_R, tc,
                                      return_cons=return_cons)
