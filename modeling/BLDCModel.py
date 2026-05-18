import numpy as np

class BLDCModel:
    def __init__(self, params, x0, dt=0.01) -> None:
        # params: [R, L, K, B, J]
        self.R, self.L, self.K, self.B, self.J = params
        self.Amat = np.array([[-self.R / self.L, -self.K / self.L], [self.K / self.J, -self.B / self.J]]).reshape(2, 2)
        self.Bmat = np.array([[1/self.L], [0]]).reshape(2, 1)
        self.dt = dt
        self.reset(x0)

    def reset(self, x0=np.zeros((2,1))) -> None:
        self.x_log = x0.reshape(2, 1)   # x: [I; omega]
        self.xdot_log = np.zeros((2, 1))    # xdot: [Idot; omega_dot]
    
    def f(self, x, v_in, tau_ext=0):
        x = x.reshape(2, 1)
        v_in = np.array(v_in).reshape(1, 1)
        Cmat = np.array([0, -tau_ext]).reshape(2, 1)
        out = (self.Amat @ x + self.Bmat @ v_in + Cmat).reshape(2, 1)
        return out
        
    def rk4(self, x, v_in, tau_ext=0):
        dt = self.dt
        x = x.reshape(2, 1)
        k1 = self.f(x, v_in, tau_ext)
        k2 = self.f(x + 0.5*dt*k1, v_in, tau_ext)
        k3 = self.f(x + 0.5*dt*k2, v_in, tau_ext)
        k4 = self.f(x + dt*k3, v_in, tau_ext)
        return (x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)).reshape(2, 1)
   
    def step(self, v_in, tau_ext=0):
        x = self.x_log[:, -1]
        x_next = self.rk4(x, v_in, tau_ext)

        # logging
        self.x_log = np.hstack((self.x_log, x_next))
        x_dot = self.f(x, v_in)
        self.xdot_log = np.hstack((self.xdot_log, x_dot))
        return x_next

    @property
    def state(self):
        x = self.x_log[:, -1]
        x_dot = self.xdot_log[:, -1]
        return np.hstack([x, x_dot])    # return 2x2 [[I; omega], [Idot; omega_dot]]