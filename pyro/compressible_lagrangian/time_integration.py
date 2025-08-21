import numpy as np
from .reconstruction import reconstruct_primitives
from .riemann import pvrs_contact
from .viscosity import vnr_q_vertical, vnr_q_horizontal
from .forces import accumulate_cell_updates
from .util import rp_get

class SSPRK2Stepper:
    def __init__(self, rp):
        self.vnr_coeff = float(rp_get(rp, "lagrangian.visc_coeff", 0.0))

    def _single_stage(self, mesh, state, bc, dt, tnow):
        prim = {'rho':state.rho, 'u':state.u, 'v':state.v, 'p':state.p}
        primx, primy = reconstruct_primitives(prim)
        ny, nx = mesh.ny, mesh.nx

        ufx = np.zeros((ny, nx+1)); pfx = np.zeros((ny, nx+1))
        for j in range(ny):
            # left boundary (could be piston)
            ufx[j,0], pfx[j,0] = bc.apply_vertical_boundary('xlb', j, state, state.gamma, tnow)
            for i in range(1, nx):
                rhoL = max(primx['rhoL'][j,i], 1e-30); rhoR = max(primx['rhoR'][j,i-1], 1e-30)
                uL   = primx['uL'][j,i];               uR   = primx['uR'][j,i-1]
                pL   = max(primx['pL'][j,i], 1e-30);   pR   = max(primx['pR'][j,i-1], 1e-30)
                ufx[j,i], pfx[j,i] = pvrs_contact(state.gamma, rhoL,uL,pL, rhoR,uR,pR)
            # right boundary
            ufx[j,nx], pfx[j,nx] = bc.apply_vertical_boundary('xrb', j, state, state.gamma, tnow)

        ufy = np.zeros((ny+1, nx)); pfy = np.zeros((ny+1, nx))
        for i in range(nx):
            ufy[0,i], pfy[0,i] = bc.apply_horizontal_boundary('ylb', i, state, state.gamma, tnow)
            for j in range(1, ny):
                rhoB = max(primy['rhoB'][j,i], 1e-30); rhoT = max(primy['rhoT'][j-1,i], 1e-30)
                vB   = primy['vB'][j,i];               vT   = primy['vT'][j-1,i]
                pB   = max(primy['pB'][j,i], 1e-30);   pT   = max(primy['pT'][j-1,i], 1e-30)
                ufy[j,i], pfy[j,i] = pvrs_contact(state.gamma, rhoB,vB,pB, rhoT,vT,pT)
            ufy[ny,i], pfy[ny,i] = bc.apply_horizontal_boundary('yrb', i, state, state.gamma, tnow)

        qx = vnr_q_vertical(mesh, state, self.vnr_coeff)
        qy = vnr_q_horizontal(mesh, state, self.vnr_coeff)
        pfx = pfx + qx; pfy = pfy + qy

        dru, drv, dE = accumulate_cell_updates(mesh, ufx, pfx, ufy, pfy, state.gamma)

        state.rho_u += dt * dru
        state.rho_v += dt * drv
        state.rho_E += dt * dE

        Un, Vn = mesh.assemble_nodal_velocity(ufx, ufy)
        mesh.move_nodes(Un, Vn, dt)

    def advance(self, mesh, state, bc, dt):
        self._single_stage(mesh, state, bc, dt, tnow=0.0)
        state.sync_primitives(mesh)
        self._single_stage(mesh, state, bc, dt, tnow=0.0)
        state.sync_primitives(mesh)
