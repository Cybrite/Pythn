import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
Cp = 4.184; rho = 1000; V = 100      # heat capacity, density, volume
U = 500; A = 10; dH = -50000         # heat transfer values
k0 = 1e3; E = 8000; R = 8.314        # reaction constants
Tf = 350                             # feed temp
Tc = 300                             # initial coolant temp
F = 1                                # flow rate
Tset = 365                           # setpoint

# PID parameters
Kp, Ki, Kd = 5, 1.2, 0.2
integral = 0
prev_error = 0

# CSTR Model
def cstr(t, T):
    global Tc, integral, prev_error
    k = k0 * np.exp(-E / (R * T))
    reaction = (-dH)*k/(rho*Cp)
    heat_in = (F/V)*(Tf - T)
    heat_transfer = U*A*(Tc - T)/(rho*Cp*V)

    # PID update
    error = Tset - T
    integral += error * 0.1
    derivative = (error - prev_error) / 0.1
    Tc = 300 + Kp*error + Ki*integral + Kd*derivative
    prev_error = error

    return heat_in + reaction + heat_transfer

# Simulate
t_span = (0, 50)
sol = solve_ivp(cstr, t_span, [330], t_eval=np.linspace(0,50,500))

# Plot
plt.plot(sol.t, sol.y[0], label='Reactor Temperature')
plt.axhline(Tset, linestyle='--', label='Setpoint')
plt.xlabel('Time (min)'); plt.ylabel('Temperature (K)')
plt.title('PID Control of CSTR Temperature')
plt.legend(); plt.grid(); plt.show()
