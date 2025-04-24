import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Given data
time = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])  # minutes
conc = np.array([0, 2, 3.5, 4.5, 5.5, 4.6, 3.4, 2.1, 0])  # g/L

# Calculate the total area under the C(t) curve
total_area = simpson(conc, time)
print(f"Total area under C(t) curve = {total_area:.2f} min·g/L")

# Calculate E(t) = C(t)/total_area
E_t = conc / total_area
print("\nE(t) values:")
for i in range(len(time)):
    print(f"t = {time[i]} min: E(t) = {E_t[i]:.5f} min⁻¹")

# Calculate mean residence time
# Method 1: Using the formula tm = ∫t·C(t)dt / ∫C(t)dt
t_Ct = time * conc
mean_residence_time = simpson(t_Ct, time) / total_area
print(f"\nMean residence time = {mean_residence_time:.2f} minutes")

# Create F(t) by integrating E(t)
F_t = np.zeros_like(time, dtype=float)
for i in range(1, len(time)):
    F_t[i] = simpson(E_t[:i+1], time[:i+1])

print("\nF(t) values:")
for i in range(len(time)):
    print(f"t = {time[i]} min: F(t) = {F_t[i]:.5f}")

# For more accurate estimation of F(12) and F(16), create interpolation functions
E_t_interp = interp1d(time, E_t, kind='cubic')
F_t_interp = interp1d(time, F_t, kind='cubic')

# Calculate the fraction spent between 12 and 16 minutes
fraction_12_to_16 = F_t_interp(16) - F_t_interp(12)
print(f"\nFraction of material that spent time between 12 and 16 minutes: {fraction_12_to_16:.5f}")

# Plot C(t), E(t), and F(t)
plt.figure(figsize=(15, 10))

# Plot C(t)
plt.subplot(3, 1, 1)
plt.plot(time, conc, 'o-', color='blue', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (min)')
plt.ylabel('Concentration (g/L)')
plt.title('C(t): Effluent Tracer Concentration')

# Plot E(t)
plt.subplot(3, 1, 2)
plt.plot(time, E_t, 'o-', color='green', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (min)')
plt.ylabel('E(t) (min⁻¹)')
plt.title('E(t): Residence Time Distribution Function')

# Plot F(t)
plt.subplot(3, 1, 3)
plt.plot(time, F_t, 'o-', color='red', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (min)')
plt.ylabel('F(t)')
plt.title('F(t): Cumulative Distribution Function')

# Highlight the region between 12 and 16 minutes on F(t) plot
t_fine = np.linspace(12, 16, 100)
F_t_fine = F_t_interp(t_fine)
plt.fill_between(t_fine, F_t_fine, F_t_interp(12) * np.ones_like(t_fine), 
                 color='red', alpha=0.3, label=f'Fraction = {fraction_12_to_16:.4f}')
plt.legend()

plt.tight_layout()
plt.show()

# Print final results
print("\nFinal Results:")
print(f"Total area under C(t) curve = {total_area:.2f} min·g/L")
print(f"Mean residence time = {mean_residence_time:.2f} minutes")
print(f"Fraction of material that spent time between 12 and 16 minutes: {fraction_12_to_16:.5f}")