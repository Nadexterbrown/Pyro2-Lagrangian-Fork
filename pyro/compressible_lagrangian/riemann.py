"""
Wrapper around Pyro's riemann.riemann_flux to add return_cons keyword.
"""

import pyro.compressible.riemann as base_riemann

def riemann_flux(dirn, q_l, q_r, *args, return_cons=False, **kwargs):
    """
    Wrapper to Pyro's riemann_flux to accept return_cons argument.
    """
    result = base_riemann.riemann_flux(dirn, q_l, q_r, *args, **kwargs)
    if return_cons:
        # Return flux and placeholder conserved state
        return result, q_l.copy()
    return result
