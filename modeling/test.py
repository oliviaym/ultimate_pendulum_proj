from BLDCModel import BLDCModel
from PendulumModel import PendulumModel
import numpy as np
import matplotlib.pyplot as plt

def test_motor():
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

def test_pendulum():
    # massless rod
    m_rod = 0.0       # kg
    m_point = 2.0     # kg
    length = 1.0     # m
    dt = 0.01  # s

    x0 = np.array([np.pi/2, 0])

    massless_pendulum = PendulumModel(m_rod, m_point, length, x0, dt)

    T = 1.0        # seconds
    steps = int(T / massless_pendulum.dt)

    for _ in range(steps):
        massless_pendulum.step()

    # Plot
    t = np.linspace(0, T, steps + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t, massless_pendulum.x_log[0, :])
    axes[0].set_ylabel('theta (rad)')
    axes[0].grid(True)
    axes[1].plot(t, massless_pendulum.x_log[1, :])
    axes[1].set_ylabel('ω (rad/s)')
    axes[1].grid(True)
    axes[1].set_xlabel('Time (s)')
    fig.suptitle('Massless Pendulum Swing')
    plt.tight_layout()
    plt.show()



test_motor()
test_pendulum()