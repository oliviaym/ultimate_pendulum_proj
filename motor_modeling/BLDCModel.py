import numpy as np
import matplotlib.pyplot as plt

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
        self.xdot_log = np.zeros((2, 1))    # xdot: [Idot, omega_dot]
    
    def f(self, x, v_in, tau_ext=0):
        x = x.reshape(2, 1)
        v_in = np.array(v_in).reshape(1, 1)
        Cmat = np.array([0, -tau_ext]).reshape(2, 1)
        return (self.Amat @ x + self.Bmat @ v_in + Cmat).reshape(2, 1)
        
    def rk4(self, x, v_in, tau_ext=0):
        dt = self.dt
        k1 = self.f(x, v_in, tau_ext)
        k2 = self.f(x + 0.5*dt*k1, v_in, tau_ext)
        k3 = self.f(x + 0.5*dt*k2, v_in, tau_ext)
        k4 = self.f(x + dt*k3, v_in, tau_ext)
        return (x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)).reshape(2, 1)
   
    def step(self, v_in, tau_ext=0):
        x = self.x_log[:, -1]
        x_dot = self.f(x, v_in)
        self.xdot_log = np.hstack((self.xdot_log, x_dot))
        x_next = self.rk4(x, v_in, tau_ext)
        self.x_log = np.hstack((self.x_log, x_next))
        return x_next
    

R = 0.5      # Ohms
L = 0.001    # Henries (1 mH)
K = 0.01     # V·s/rad (back-EMF and torque constant)
B = 0.0001   # N·m·s/rad (friction)
J = 0.00001  # kg·m² (rotor inertia)

params = [R, L, K, B, J]
x0 = np.zeros((2, 1))
motor = BLDCModel(params, x0, dt=0.001)

V_step = 12.0  # volts
T = 1.0        # seconds
steps = int(T / motor.dt)

for _ in range(steps):
    motor.step(V_step)

# Plot
t = np.linspace(0, T, steps + 1)
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t, motor.x_log[0, :])
axes[0].set_ylabel('Current I (A)')
axes[0].grid(True)
axes[1].plot(t, motor.x_log[1, :])
axes[1].set_ylabel('ω (rad/s)')
axes[1].grid(True)
axes[1].set_xlabel('Time (s)')
fig.suptitle('Motor Step Response')
plt.tight_layout()
plt.show()