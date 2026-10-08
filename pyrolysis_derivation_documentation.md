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

- y = [m_c, m_h, m_l, m_v, m_ch, m_g, T]

where:

- m_c = mass of cellulose (kg)
- m_h = mass of hemicellulose (kg)
- m_l = mass of lignin (kg)
- m_v = mass of vapor (kg)
- m_ch = mass of char (kg)
- m_g = mass of gas (kg)
- T = reactor temperature (K)

The total initial biomass mass is:

m0 = 0.10 kg

The biomass composition fractions are:

- f_c = 0.45
- f_h = 0.30
- f_l = 0.25

So the initial masses are:

m_c0 = m0 × f_c = 0.10 × 0.45 = 0.045 kg
m_h0 = m0 × f_h = 0.10 × 0.30 = 0.030 kg
m_l0 = m0 × f_l = 0.10 × 0.25 = 0.025 kg

All product masses start at zero:

m_v0 = m_ch0 = m_g0 = 0

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

k_c = 70000 × exp(-80000 / (R × T))
k_h = 2e9 × exp(-146000 / (R × T))
k_l = 4300 × exp(-77000 / (R × T))
k_s = 2e5 × exp(-95000 / (R × T))

These are the decomposition rates of:

- cellulose,
- hemicellulose,
- lignin,
- secondary cracking of vapor.

The model assumes the decomposition rate is proportional to the remaining mass of each reactive species:

r_c = k_c × m_c
r_h = k_h × m_h
r_l = k_l × m_l
r_s = k_s × m_v

These are first-order mass-loss reactions.

---

## 4. Derivation of the biomass mass balances

### 4.1 Cellulose balance

Cellulose decomposes directly and is consumed according to its reaction rate:

d(m_c)/dt = -r_c

This matches the code:

dm_c_dt = -r_c

### 4.2 Hemicellulose balance

Similarly:

d(m_h)/dt = -r_h

### 4.3 Lignin balance

Similarly:

d(m_l)/dt = -r_l

### 4.4 Vapor balance

The vapor is formed from primary decomposition and reduced by secondary cracking:

d(m_v)/dt = 0.78 r_c + 0.70 r_h + 0.35 r_l - r_s

This is exactly the code expression:

dm_v_dt = 0.78*r_c + 0.70*r_h + 0.35\*r_l - r_s

Interpretation:

- 78% of cellulose conversion becomes vapor
- 70% of hemicellulose conversion becomes vapor
- 35% of lignin conversion becomes vapor
- secondary vapor cracking consumes vapor at rate r_s

### 4.5 Char balance

Char is produced from decomposition and also from secondary cracking:

d(m_ch)/dt = 0.10 r_c + 0.15 r_h + 0.45 r_l + 0.25 r_s

This is the code line:

dm_ch_dt = 0.10*r_c + 0.15*r_h + 0.45*r_l + 0.25*r_s

### 4.6 Gas balance

Gas is produced from both primary decomposition and secondary cracking:

d(m_g)/dt = 0.12 r_c + 0.15 r_h + 0.20 r_l + 0.75 r_s

This matches the code:

dm_g_dt = 0.12*r_c + 0.15*r_h + 0.20*r_l + 0.75*r_s

---

## 5. Mass conservation check

The total mass is:

m_total = m_c + m_h + m_l + m_v + m_ch + m_g

Its time derivative is:

d(m_total)/dt = d(m_c)/dt + d(m_h)/dt + d(m_l)/dt + d(m_v)/dt + d(m_ch)/dt + d(m_g)/dt

Substituting the expressions above:

d(m_total)/dt = -r_c - r_h - r_l + (0.78 r_c + 0.70 r_h + 0.35 r_l - r_s) + (0.10 r_c + 0.15 r_h + 0.45 r_l + 0.25 r_s) + (0.12 r_c + 0.15 r_h + 0.20 r_l + 0.75 r_s)

Grouping terms:

- cellulose contribution: -1 + 0.78 + 0.10 + 0.12 = 0
- hemicellulose contribution: -1 + 0.70 + 0.15 + 0.15 = 0
- lignin contribution: -1 + 0.35 + 0.45 + 0.20 = 0
- secondary cracking contribution: -1 + 0.25 + 0.75 = 0

So:

d(total_mass)/dt = 0

Therefore the model conserves total mass exactly, as expected.

This is one of the strongest points of the code.

---

## 6. Derivation of the energy equation

The temperature equation in the code is:

q_in = UA × (T_w - T)

q_rxn = dH × (r_c + r_h + r_l)

dT_dt = (q_in - q_rxn) / (m0 × Cp)

where:

- UA = heat transfer coefficient
- T_w = reactor wall temperature
- T = biomass temperature
- dH = reaction enthalpy
- Cp = biomass heat capacity

### 6.1 Wall temperature profile

The code defines:

T_w = min(T0 + 10t, Tw_max)

with:

- T0 = 298.15 K
- Tw_max = 773.15 K

This means the wall is heated linearly at 10 K/s until it reaches a maximum of 773.15 K.

### 6.2 Energy balance

A lumped energy balance is approximated as:

m0 × Cp × dT/dt = UA (T_w - T) - dH (r_c + r_h + r_l)

Rearranging:

dT/dt = [UA (T_w - T) - dH (r_c + r_h + r_l)] / (m0 × Cp)

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
   - The code uses biomass_initial_mass instead of the time-varying total mass.
   - A more rigorous model would use total_mass(t) in the energy term.

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
