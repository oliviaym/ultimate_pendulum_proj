import numpy as np

class PendulumModel:
    def __init__(self, rod_mass, point_mass, length, x0, dt=0.01):
        self.m_r = rod_mass
        self.m_p = point_mass
        self.l = length

        self.J_tot = 1/3*self.m_r*self.l**2 + self.m_p*self.l**2

        self.g = 9.81
        self.dt = dt

        self.reset(x0)

    def reset(self, x0=np.zeros((2,1))):
        self.x_log = x0.reshape(2, 1)   # x: [theta; theta_dot]
        self.xdot_log = np.zeros((2, 1))    # xdot: [theta_dot; theta_ddot]

    def f(self, x, tau=0):
        p = x[0]
        v = x[1]
        g = self.g

        p_dot = v
        v_dot = (tau - self.m_p*g*self.l*np.sin(p) - self.m_r*g*self.l*np.sin(p)/2) / self.J_tot

        return np.array([p_dot, v_dot]).reshape(2, 1)
    
    def rk4(self, x, tau=0):
        dt = self.dt
        x = x.reshape(2, 1)
        k1 = self.f(x, tau)
        k2 = self.f(x + 0.5*dt*k1, tau)
        k3 = self.f(x + 0.5*dt*k2, tau)
        k4 = self.f(x + dt*k3, tau)
        return (x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)).reshape(2, 1)

    def step(self, tau=0):
        x = self.x_log[:, -1]
        x_next = self.rk4(x, tau)

        pos = (x_next[0] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        vel = x_next[1]

        x_next = np.array([pos, vel]).reshape(2, 1)

        # logging
        self.x_log = np.hstack((self.x_log, x_next))
        x_dot = self.f(x, tau)
        self.xdot_log = np.hstack((self.xdot_log, x_dot))

        return x_next

    @property
    def state(self):
        x = self.x_log[:, -1]
        return x    # return 2x1 vector [theta; theta_dot]
    
class PendulumMotorSystem:
    def __init__(self, pendulum, motor, x0, dt=0.01):
        self.pendulum = pendulum
        self.motor = motor
        self.dt = dt
        # combined state: [I, theta, theta_dot]
        self.reset(x0)

    def reset(self, x0=np.zeros((3, 1))):
        self.x_log = x0.reshape(3, 1)   # x: [I; theta; theta_dot]
        self.xdot_log = np.zeros((3, 1))    # xdot: [I_dot; theta_dot; theta_ddot]
    
    def f(self, x, v_in):
        I, theta, theta_dot = x.flatten()
        
        # motor electrical equation (owns I dynamics)
        R, L, K, B, J_mot = self.motor.R, self.motor.L, self.motor.K, self.motor.B, self.motor.J
        I_dot = (v_in - K*theta_dot - I * R) / L
        
        # combined mechanical equation (owns theta dynamics)
        m_p = self.pendulum.m_p
        m_r = self.pendulum.m_r
        g = self.pendulum.g
        l = self.pendulum.l
        theta_ddot = (K*I - m_p*g*l*np.sin(theta) - m_r*g*l*np.sin(theta)/2 - B*theta_dot) / (self.pendulum.J_tot + self.motor.J)
        
        return np.array([I_dot, theta_dot, theta_ddot]).reshape(3, 1)
    
    def rk4(self, x, v_in):
        dt = self.dt
        x = x.reshape(3, 1)
        k1 = self.f(x, v_in)
        k2 = self.f(x + 0.5*dt*k1, v_in)
        k3 = self.f(x + 0.5*dt*k2, v_in)
        k4 = self.f(x + dt*k3, v_in)
        return (x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)).reshape(3, 1)
    
    def step(self, v_in):
        x = self.x_log[:, -1]
        x_next = self.rk4(x, v_in)
        x_next[1] = (x_next[1] + np.pi) % (2 * np.pi) - np.pi

        # logging
        self.x_log = np.hstack((self.x_log, x_next))
        x_dot = self.f(x, v_in)
        self.xdot_log = np.hstack((self.xdot_log, x_dot))
        return x_next
    
    @property
    def state(self):
        x = self.x_log[:, -1]
        return x    # return 3x1 vector [I, theta, theta_dot]