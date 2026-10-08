import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# PARAMETERS
# ============================================================

m0 = 0.10                 # Initial biomass mass (kg)
T_in = 298.15             # Initial temperature (K)
Cp_bio = 1600             # Biomass heat capacity (J/kg-K)
UA = 8                    # Heat transfer coefficient (W/K)
dH_pyr = 300000           # Pyrolysis enthalpy (J/kg)
T_w_max = 773.15          # Maximum wall temperature (K)
R = 8.314                 # Gas constant (J/mol-K)

# Biomass composition fractions
f_cell = 0.45
f_hem = 0.30
f_lig = 0.25

# ============================================================
# INITIAL CONDITIONS
# ============================================================

m_cell_0 = m0 * f_cell
m_hem_0 = m0 * f_hem
m_lig_0 = m0 * f_lig

m_vap_0 = 0
m_char_0 = 0
m_gas_0 = 0

# State vector:
# y = [cellulose, hemicellulose, lignin,
#      vapor, char, gas, temperature]

y0 = [
    m_cell_0,
    m_hem_0,
    m_lig_0,
    m_vap_0,
    m_char_0,
    m_gas_0,
    T_in
]

# ============================================================
# ODE FUNCTION
# ============================================================

def pyrolysis_odes(t, y):

    # State variables
    m_cell = y[0]
    m_hem = y[1]
    m_lig = y[2]
    m_vap = y[3]
    m_char = y[4]
    m_gas = y[5]
    T = y[6]

    # --------------------------------------------------------
    # Wall temperature
    # --------------------------------------------------------

    T_w = min(T_in + 10*t, T_w_max)

    # --------------------------------------------------------
    # Reaction rate constants
    # --------------------------------------------------------

    k_cell = 70000 * np.exp(-80000 / (R*T))

    k_hem = 2e9 * np.exp(-146000 / (R*T))

    k_lignin = 4300 * np.exp(-77000 / (R*T))

    k_secondary = 2e5 * np.exp(-95000 / (R*T))

    # --------------------------------------------------------
    # Reaction rates
    # --------------------------------------------------------

    r_cell = k_cell * m_cell

    r_hem = k_hem * m_hem

    r_lignin = k_lignin * m_lig

    r_S = k_secondary * m_vap

    # --------------------------------------------------------
    # ODEs
    # --------------------------------------------------------

    dm_cell_dt = -r_cell

    dm_hem_dt = -r_hem

    dm_lig_dt = -r_lignin

    # Vapor
    dm_vap_dt = (
        0.78*r_cell
        + 0.70*r_hem
        + 0.35*r_lignin
        - r_S
    )

    # Char
    dm_char_dt = (
        0.10*r_cell
        + 0.15*r_hem
        + 0.45*r_lignin
        + 0.25*r_S
    )

    # Gas
    dm_gas_dt = (
        0.12*r_cell
        + 0.15*r_hem
        + 0.20*r_lignin
        + 0.75*r_S
    )

    # --------------------------------------------------------
    # Energy balance
    # --------------------------------------------------------

    heat_transfer = UA * (T_w - T)

    heat_required = dH_pyr * (
        r_cell + r_hem + r_lignin
    )

    dT_dt = (
        heat_transfer - heat_required
    ) / (m0 * Cp_bio)

    # Return derivatives
    dydt = [
        dm_cell_dt,
        dm_hem_dt,
        dm_lig_dt,
        dm_vap_dt,
        dm_char_dt,
        dm_gas_dt,
        dT_dt
    ]

    return dydt


# ============================================================
# SOLVE ODEs
# ============================================================

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

m_cell = y[0]
m_hem = y[1]
m_lig = y[2]
m_vap = y[3]
m_char = y[4]
m_gas = y[5]
T = y[6]


# ============================================================
# GRAPH 1: MASS PROFILES
# ============================================================

plt.figure(figsize=(9,6))

plt.plot(t, m_cell, label='Cellulose', linewidth=2)
plt.plot(t, m_hem, label='Hemicellulose', linewidth=2)
plt.plot(t, m_lig, label='Lignin', linewidth=2)
plt.plot(t, m_vap, label='Vapor', linewidth=2)
plt.plot(t, m_char, label='Char', linewidth=2)
plt.plot(t, m_gas, label='Gas', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.ylim(0, m0)
plt.legend()
plt.grid(True)

plt.title('Biomass Pyrolysis Mass Profiles')

plt.show()


# ============================================================
# GRAPH 2: TEMPERATURE PROFILE
# ============================================================

plt.figure(figsize=(9,6))

plt.plot(t, T, 'k-', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Biomass Temperature Profile')

plt.grid(True)

plt.show()


# ============================================================
# GRAPH 3: WALL TEMPERATURE
# ============================================================

T_wall = np.minimum(T_in + 10*t, T_w_max)

plt.figure(figsize=(9,6))

plt.plot(t, T_wall, linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Wall Temperature (K)')
plt.title('Wall Temperature Profile')

plt.grid(True)

plt.show()


# ============================================================
# MASS CONSERVATION CHECK
# ============================================================

m_total = (
    m_cell
    + m_hem
    + m_lig
    + m_vap
    + m_char
    + m_gas
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
