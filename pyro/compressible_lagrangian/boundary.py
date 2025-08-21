class PistonBC:
    def __init__(self,eos,gamma,piston_vel_fn,rho_floor=1e-16,p_floor=1e-16,a_floor=1e-16):
        self.eos=eos; self.gamma=gamma; self.piston_vel_fn=piston_vel_fn
        self.rho_floor=rho_floor; self.p_floor=p_floor; self.a_floor=a_floor
    def ghost_left(self,prim_cell,t):
        tau_i,u_i,p_i = prim_cell["tau"], prim_cell["u"], prim_cell["p"]
        rho_i = max(1.0/max(tau_i, 1.0/self.rho_floor), self.rho_floor)
        a_i = max((self.gamma*p_i/rho_i)**0.5, self.a_floor)
        K = p_i/(rho_i**self.gamma); Jm = u_i - 2.0*a_i/(self.gamma-1.0)
        u_b=float(self.piston_vel_fn(t))
        a_b=max(0.5*(self.gamma-1.0)*(u_b - Jm), self.a_floor)
        rho_b = ((a_b*a_b)/(self.gamma*max(K,self.p_floor)))**(1.0/(self.gamma-1.0))
        rho_b = max(rho_b, self.rho_floor)
        p_b = K*(rho_b**self.gamma); tau_b=1.0/rho_b
        return tau_b, u_b, p_b
