import numpy as np

def limited_slope(aL, aC, aR, limiter='minmod'):
    dl = aC - aL
    dr = aR - aC
    if limiter == 'minmod':
        s = np.sign(dl) + np.sign(dr)
        return 0.5 * s * np.minimum(np.abs(dl), np.abs(dr))
    if limiter == 'mc':
        return np.minimum.reduce([2*np.abs(dl), 2*np.abs(dr),
                                  0.5*np.abs(dl+dr)]) * np.sign(dl+dr)
    return 0.0

def reconstruct_primitives(prim):
    """Very simple 1D-in-each-direction MUSCL slopes.
    prim: dict with keys 'rho','u','v','p' each (ny,nx).
    Returns left/right/top/bottom interface values as dict.
    """
    rho = prim['rho']; u=prim['u']; v=prim['v']; p=prim['p']
    ny, nx = rho.shape
    # x-direction (vertical faces): states at i-1/2
    def rec_x(A):
        L = np.zeros((ny, nx+1))
        R = np.zeros((ny, nx+1))
        for j in range(ny):
            for i in range(1, nx):
                s = limited_slope(A[j,i-1], A[j,i], A[j,min(i+1,nx-1)])
                R[j,i-1] = A[j,i-1] + 0.5*s
                L[j,i]   = A[j,i]   - 0.5*s
            # boundary copy
            L[j,0] = A[j,0]; R[j,nx-1] = A[j,nx-1]
        return L, R
    # y-direction (horizontal faces): states at j-1/2
    def rec_y(A):
        B = np.zeros((ny+1, nx))
        T = np.zeros((ny+1, nx))
        for i in range(nx):
            for j in range(1, ny):
                s = limited_slope(A[j-1,i], A[j,i], A[min(j+1,ny-1),i])
                T[j-1,i] = A[j-1,i] + 0.5*s
                B[j,i]   = A[j,i]   - 0.5*s
            B[0,i] = A[0,i]; T[ny-1,i] = A[ny-1,i]
        return B, T
    primx = {}
    primy = {}
    for key,A in (('rho',rho),('u',u),('v',v),('p',p)):
        L,R = rec_x(A); primx[key+'L']=L; primx[key+'R']=R
        B,T = rec_y(A); primy[key+'B']=B; primy[key+'T']=T
    return primx, primy
