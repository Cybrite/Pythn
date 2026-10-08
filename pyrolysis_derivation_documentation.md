# Biomass Pyrolysis Model: Derivation and Evaluation

## 1. Model purpose

This script models a lumped biomass pyrolysis process in which the feed is represented as three reactive solids:

- cellulose
- hemicellulose
- lignin

These solids decompose to produce:

- vapor
- char
- gas

The model also tracks temperature evolution during heating and reaction.

---

## 2. State variables and initial conditions

The state vector is:

- y = [m_cell, m_hem, m_lig, m_vap, m_char, m_gas, T]

where:

- m_cell = mass of cellulose (kg)
- m_hem = mass of hemicellulose (kg)
- m_lig = mass of lignin (kg)
- m_vap = mass of vapor (kg)
- m_char = mass of char (kg)
- m_gas = mass of gas (kg)
- T = reactor temperature (K)

The total initial biomass mass is:

m0 = 0.10 kg

The biomass composition fractions are:

- f_cell = 0.45
- f_hem = 0.30
- f_lig = 0.25

So the initial masses are:

m_cell,0 = m0 _ f_cell = 0.10 _ 0.45 = 0.045 kg
m_hem,0 = m0 _ f_hem = 0.10 _ 0.30 = 0.030 kg
m_lig,0 = m0 _ f_lig = 0.10 _ 0.25 = 0.025 kg

All product masses start at zero:

m_vap,0 = m_char,0 = m_gas,0 = 0

---

## 3. Reaction kinetics

Each biomass component follows an Arrhenius-type rate law:

k_i(T) = A_i exp(-E_i / (R T))

where:

- A_i = pre-exponential factor
- E_i = activation energy
- R = gas constant = 8.314 J/mol-K
- T = temperature in K

The code uses:

k_cell = 70000 _ exp(-80000 / (R T))
k_hem = 2e9 _ exp(-146000 / (R T))
k_lignin = 4300 _ exp(-77000 / (R T))
k_secondary = 2e5 _ exp(-95000 / (R T))

These are the decomposition rates of:

- cellulose,
- hemicellulose,
- lignin,
- secondary cracking of vapor.

The model assumes the decomposition rate is proportional to the remaining mass of each reactive species:

r_cell = k_cell _ m_cell
r_hem = k_hem _ m_hem
r_lignin = k_lignin _ m_lig
r_S = k_secondary _ m_vap

These are first-order mass-loss reactions.

---

## 4. Derivation of the biomass mass balances

### 4.1 Cellulose balance

Cellulose decomposes directly and is consumed according to its reaction rate:

dm_cell/dt = -r_cell

This matches the code:

dm_cell_dt = -r_cell

### 4.2 Hemicellulose balance

Similarly:

dm_hem/dt = -r_hem

### 4.3 Lignin balance

Similarly:

dm_lig/dt = -r_lignin

### 4.4 Vapor balance

The vapor is formed from primary decomposition and reduced by secondary cracking:

dm_vap/dt = 0.78 r_cell + 0.70 r_hem + 0.35 r_lignin - r_S

This is exactly the code expression:

dm_vap_dt = 0.78*r_cell + 0.70*r_hem + 0.35\*r_lignin - r_S

Interpretation:

- 78% of cellulose conversion becomes vapor
- 70% of hemicellulose conversion becomes vapor
- 35% of lignin conversion becomes vapor
- secondary vapor cracking consumes vapor at rate r_S

### 4.5 Char balance

Char is produced from decomposition and also from secondary cracking:

dm_char/dt = 0.10 r_cell + 0.15 r_hem + 0.45 r_lignin + 0.25 r_S

This is the code line:

dm_char_dt = 0.10*r_cell + 0.15*r_hem + 0.45*r_lignin + 0.25*r_S

### 4.6 Gas balance

Gas is produced from both primary decomposition and secondary cracking:

dm_gas/dt = 0.12 r_cell + 0.15 r_hem + 0.20 r_lignin + 0.75 r_S

This matches the code:

dm_gas_dt = 0.12*r_cell + 0.15*r_hem + 0.20*r_lignin + 0.75*r_S

---

## 5. Mass conservation check

The total mass is:

m_total = m_cell + m_hem + m_lig + m_vap + m_char + m_gas

Its time derivative is:

dm_total/dt = dm_cell/dt + dm_hem/dt + dm_lig/dt + dm_vap/dt + dm_char/dt + dm_gas/dt

Substituting the expressions above:

dm_total/dt = -r_cell - r_hem - r_lignin + (0.78r_cell + 0.70r_hem + 0.35r_lignin - r_S) + (0.10r_cell + 0.15r_hem + 0.45r_lignin + 0.25r_S) + (0.12r_cell + 0.15r_hem + 0.20r_lignin + 0.75r_S)

Grouping terms:

- cellulose contribution: -1 + 0.78 + 0.10 + 0.12 = 0
- hemicellulose contribution: -1 + 0.70 + 0.15 + 0.15 = 0
- lignin contribution: -1 + 0.35 + 0.45 + 0.20 = 0
- secondary cracking contribution: -1 + 0.25 + 0.75 = 0

So:

dm_total/dt = 0

Therefore the model conserves total mass exactly, as expected.

This is one of the strongest points of the code.

---

## 6. Derivation of the energy equation

The temperature equation in the code is:

heat_transfer = UA \* (T_w - T)

heat_required = dH_pyr \* (r_cell + r_hem + r_lignin)

dT_dt = (heat_transfer - heat_required) / (m0 \* Cp_bio)

where:

- UA = heat transfer coefficient
- T_w = wall temperature
- T = biomass temperature
- dH_pyr = pyrolysis enthalpy
- Cp_bio = biomass heat capacity

### 6.1 Wall temperature profile

The code defines:

T_w = min(T_in + 10 t, T_w_max)

with:

- T_in = 298.15 K
- T_w_max = 773.15 K

This means the wall is heated linearly at 10 K/s until it reaches a maximum of 773.15 K.

### 6.2 Energy balance

A lumped energy balance is approximated as:

m0 _ Cp_bio _ dT/dt = UA (T_w - T) - dH_pyr (r_cell + r_hem + r_lignin)

Rearranging:

dT/dt = [UA (T_w - T) - dH_pyr (r_cell + r_hem + r_lignin)] / (m0 \* Cp_bio)

This is the code’s exact temperature balance.

Interpretation:

- heat enters from the wall to the biomass
- heat is consumed by endothermic pyrolysis reactions
- the model assumes uniform temperature throughout the biomass lump
- the mass is treated as constant for the heat-capacity term

---

## 7. Evaluation of the code

### Strengths

- The model is structurally consistent and mass-conservative.
- The kinetic terms are physically plausible Arrhenius forms.
- The state equations are easy to interpret and solve using solve_ivp.
- The mass-yield coefficients are consistent with a pyrolysis product distribution.

### Numerical verification

The model was checked numerically:

- initial total mass = 0.100000000000 kg
- final total mass = 0.100000000000 kg
- maximum mass error = 4.163336342344337e-17 kg

This confirms the code is conserving mass to numerical precision.

### Limitations and conceptual simplifications

1. Constant total mass in the denominator:
   - The code uses m0 instead of the time-varying total mass.
   - A more rigorous model would use m_total(t) in the energy term.

2. Single lumped temperature:
   - It assumes all biomass and products are at the same temperature.
   - In reality, internal temperature gradients may exist.

3. No moisture or drying step:
   - The model starts directly with dry biomass.

4. No particle size or transport effects:
   - There is no heat conduction, diffusion, or reaction-diffusion inside the solid.

5. Fixed yield coefficients:
   - The conversion fractions are constants rather than temperature-dependent yields.

6. Simplified reactor wall heating:
   - wall temperature is imposed as a ramp, not calculated from a full heat-transfer model.

---

## 8. Conclusion

The code is a reasonable lumped kinetic pyrolysis model for educational and conceptual analysis. It captures the main ideas of:

- Arrhenius decomposition kinetics,
- product yield allocation,
- mass conservation,
- heat balance with wall heating and endothermic reaction heat.

The equations are internally consistent, and the implementation numerically preserves mass. The main improvement would be to refine the thermal model to include time-varying mass and more realistic thermal transport behavior.

---

## 9. Suggested engineering interpretation

In practical terms, the model describes a reactor where the wall is heated, the biomass decomposes into vapor, char, and gas, and the endothermic heat of reaction competes with heat input from the wall. The mass fractions determine how each component splits into products, while the Arrhenius terms control how quickly decomposition accelerates as temperature rises.
