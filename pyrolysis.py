import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# PARAMETERS

m0 = 0.10                 # Initial biomass mass (kg)
T0 = 298.15               # Initial temperature (K)
Cp = 1600                 # Biomass heat capacity (J/kg-K)
UA = 8                    # Heat transfer coefficient (W/K)
dH = 300000               # Pyrolysis enthalpy (J/kg)
Tw_max = 773.15           # Maximum wall temperature (K)
R = 8.314                 # Gas constant (J/mol-K)

# Biomass composition fractions
f_c = 0.45
f_h = 0.30
f_l = 0.25

# INITIAL CONDITIONS

m_c0 = m0 * f_c
m_h0 = m0 * f_h
m_l0 = m0 * f_l

m_v0 = 0
m_ch0 = 0
m_g0 = 0

# State vector:
# y = [cellulose, hemicellulose, lignin,
#      vapor, char, gas, temperature]

y0 = [
    m_c0,
    m_h0,
    m_l0,
    m_v0,
    m_ch0,
    m_g0,
    T0
]

# ODE FUNCTION

def pyrolysis_odes(t, y):

    # State variables for biomass and products
    m_c = y[0]  # Cellulose mass
    m_h = y[1]  # Hemicellulose mass
    m_l = y[2]  # Lignin mass
    m_v = y[3]  # Vapor mass
    m_ch = y[4] # Char mass
    m_g = y[5]  # Gas mass
    T = y[6]    # Bulk temperature

    # Wall temperature profile
    T_w = min(T0 + 10*t, Tw_max)

    # Reaction rate constants
    k_c = 70000 * np.exp(-80000 / (R*T))
    k_h = 2e9 * np.exp(-146000 / (R*T))
    k_l = 4300 * np.exp(-77000 / (R*T))
    k_s = 2e5 * np.exp(-95000 / (R*T))

    # Conversion rates for each reaction step
    r_c = k_c * m_c # Cellulose decomposition rate
    r_h = k_h * m_h  # Hemicellulose decomposition rate
    r_l = k_l * m_l  # Lignin decomposition rate
    r_s = k_s * m_v  # Secondary vapor cracking rate

    # ODEs
    dm_c_dt = -r_c  # Cellulose consumption
    dm_h_dt = -r_h  # Hemicellulose consumption
    dm_l_dt = -r_l  # Lignin consumption

    # Vapor evolution
    dm_v_dt = (
        0.78*r_c
        + 0.70*r_h
        + 0.35*r_l
        - r_s
    )

    # Char formation
    dm_ch_dt = (
        0.10*r_c
        + 0.15*r_h
        + 0.45*r_l
        + 0.25*r_s
    )

    # Gas formation
    dm_g_dt = (
        0.12*r_c
        + 0.15*r_h
        + 0.20*r_l
        + 0.75*r_s
    )

    # Energy balance for the lumped biomass
    q_in = UA * (T_w - T)  # Heat supplied from wall to biomass
    q_rxn = dH * (r_c + r_h + r_l)  # Heat consumed by pyrolysis reactions
    dT_dt = (
        q_in - q_rxn
    ) / (m0 * Cp)

    # Return the derivative vector
    dydt = [
        dm_c_dt,
        dm_h_dt,
        dm_l_dt,
        dm_v_dt,
        dm_ch_dt,
        dm_g_dt,
        dT_dt
    ]

    return dydt


# SOLVE ODEs

t_start = 0
t_end = 300

t_eval = np.linspace(t_start, t_end, 1000)

solution = solve_ivp(
    pyrolysis_odes,
    [t_start, t_end],
    y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# Extract results
t = solution.t
y = solution.y

m_c = y[0]
m_h = y[1]
m_l = y[2]
m_v = y[3]
m_ch = y[4]
m_g = y[5]
T = y[6]


#Temperature profile of biomass
plt.figure(figsize=(9,6))

plt.plot(t, T, 'k-', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Biomass Temperature Profile')

plt.grid(True)

plt.show()

# Wall temperature profile
T_wall = np.minimum(T0 + 10*t, Tw_max)

plt.figure(figsize=(9,6))

plt.plot(t, T_wall, linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Wall Temperature (K)')
plt.title('Wall Temperature Profile')

plt.grid(True)

plt.show()

# Mass profiles of biomass and products
plt.figure(figsize=(9,6))

plt.plot(t, m_c, label='Cellulose', linewidth=2)
plt.plot(t, m_h, label='Hemicellulose', linewidth=2)
plt.plot(t, m_l, label='Lignin', linewidth=2)
plt.plot(t, m_v, label='Vapor', linewidth=2)
plt.plot(t, m_ch, label='Char', linewidth=2)
plt.plot(t, m_g, label='Gas', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.ylim(0, m0)
plt.legend()
plt.grid(True)

plt.title('Biomass Pyrolysis Mass Profiles')

plt.show()



# Mass conservation check
m_total = (
    m_c
    + m_h
    + m_l
    + m_v
    + m_ch
    + m_g
)

plt.figure(figsize=(9,6))

plt.plot(t, m_total, linewidth=2)
plt.axhline(m0, linestyle='--', linewidth=1.5)

plt.xlabel('Time (s)')
plt.ylabel('Total Mass (kg)')
plt.title('Mass Conservation Check')

plt.grid(True)

plt.show()

print("Initial total mass =", m_total[0], "kg")
print("Final total mass   =", m_total[-1], "kg")
print("Maximum mass error  =", np.max(np.abs(m_total - m0)), "kg")
