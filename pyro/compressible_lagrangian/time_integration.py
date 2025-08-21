import numpy as np
from .reconstruction import reconstruct_primitives
from .riemann import pvrs_contact
from .viscosity import vnr_q_vertical, vnr_q_horizontal
from .forces import accumulate_cell_updates

class SSPRK2Stepper:
    def __init__(self, rp):
        self.vnr_coeff = float(rp.get_param("lagrangian.visc_coeff", 0.0))

    def _single_stage(self, mesh, state, bc, dt):
        # reconstruct
        prim = {'rho':state.rho, 'u':state.u, 'v':state.v, 'p':state.p}
        primx, primy = reconstruct_primitives(prim)
        ny, nx = mesh.ny, mesh.nx

        # vertical faces (ny, nx+1): compute u*, p* with PVRS
        ufx = np.zeros((ny, nx+1))
        pfx = np.zeros((ny, nx+1))
        for j in range(ny):
            # left boundary
            ufx[j,0], pfx[j,0] = bc.apply_vertical_boundary('xlb', j, state, state.gamma, 0.0)
            # interior faces
            for i in range(1, nx):
                rhoL = np.maximum(primx['rhoL'][j,i], 1e-30)
                rhoR = np.maximum(primx['rhoR'][j,i-1], 1e-30)
                uL   = primx['uL'][j,i]
                uR   = primx['uR'][j,i-1]
                pL   = np.maximum(primx['pL'][j,i], 1e-30)
                pR   = np.maximum(primx['pR'][j,i-1], 1e-30)
                ufx[j,i], pfx[j,i] = pvrs_contact(state.gamma, rhoL,uL,pL, rhoR,uR,pR)
            # right boundary
            ufx[j,nx], pfx[j,nx] = bc.apply_vertical_boundary('xrb', j, state, state.gamma, 0.0)

        # horizontal faces (ny+1, nx)
        ufy = np.zeros((ny+1, nx))
        pfy = np.zeros((ny+1, nx))
        for i in range(nx):
            ufy[0,i], pfy[0,i] = bc.apply_horizontal_boundary('ylb', i, state, state.gamma, 0.0)
            for j in range(1, ny):
                rhoB = np.maximum(primy['rhoB'][j,i], 1e-30)
                rhoT = np.maximum(primy['rhoT'][j-1,i], 1e-30)
                vB   = primy['vB'][j,i]
                vT   = primy['vT'][j-1,i]
                pB   = np.maximum(primy['pB'][j,i], 1e-30)
                pT   = np.maximum(primy['pT'][j-1,i], 1e-30)
                ufy[j,i], pfy[j,i] = pvrs_contact(state.gamma, rhoB,vB,pB, rhoT,vT,pT)
            ufy[ny,i], pfy[ny,i] = bc.apply_horizontal_boundary('yrb', i, state, state.gamma, 0.0)

        # artificial viscosity
        qx = vnr_q_vertical(mesh, state, self.vnr_coeff)
        qy = vnr_q_horizontal(mesh, state, self.vnr_coeff)
        pfx = pfx + qx
        pfy = pfy + qy

        # accumulate pressure forces & work into updates
        dru, drv, dE = accumulate_cell_updates(mesh, ufx, pfx, ufy, pfy, state.gamma)

        # update conserved vars (mass is constant per cell in pure Lagrangian)
        state.rho_u += dt * dru
        state.rho_v += dt * drv
        state.rho_E += dt * dE

        # move nodes with face star speeds (assemble nodal vel from faces)
        Un, Vn = mesh.assemble_nodal_velocity(ufx, ufy)
        mesh.move_nodes(Un, Vn, dt)

    def advance(self, mesh, state, bc, dt):
        # Stage 1
        self._single_stage(mesh, state, bc, dt)
        state.sync_primitives(mesh)
        # Stage 2 (Heun)
        self._single_stage(mesh, state, bc, dt)
        state.sync_primitives(mesh)
