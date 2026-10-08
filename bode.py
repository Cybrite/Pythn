import numpy as np
import matplotlib.pyplot as plt

# Define the numerator and denominator polynomials
# G_OL(s) = K * (s+3) / (s^2 + 4s + 5)
num = [1, 3]
den = [1, 4, 5]

# Compute root locus manually for K values (closed-loop char poly: D(s) + K*N(s))
k_values = np.linspace(0, 50, 5000)
# Ensure numerator is left-padded to match denominator degree
pad_len = len(den) - len(num)
num_padded = np.pad(num, (pad_len, 0), 'constant') if pad_len >= 0 else np.pad(num, (0, -pad_len), 'constant')
rlist = []
for K in k_values:
    # characteristic polynomial coefficients: den + K * num_padded
    char_poly = np.array(den, dtype=float) + K * np.array(num_padded, dtype=float)
    roots = np.roots(char_poly)
    rlist.append(roots)
rlist = np.array(rlist)  # shape: (len(k_values), n_roots)

plt.figure(figsize=(10, 6))

# Plot the root locus branches
# transpose so each column (a root across K) is plotted as a branch
plt.plot(rlist.real.T, rlist.imag.T, 'b', label='Root Locus')

# Plot the open-loop poles (x)
poles = np.roots(den)
plt.plot(poles.real, poles.imag, 'rx', markersize=10, label='Poles ($K=0$)')

# Plot the open-loop zeros (o)
zeros = np.roots(num)
plt.plot(zeros.real, zeros.imag, 'go', markersize=8, label='Zeros ($K=\infty$)')

# Add the valid break-in point for visual clarity
break_in_point = -3 - np.sqrt(2)
plt.plot(break_in_point, 0, 'ms', markersize=8, label='Break-in Point')

# Set labels and title
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.title('Root Locus Diagram for $G_{OL}(s) = K \\frac{s+3}{s^2 + 4s + 5}$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal') # Maintain aspect ratio
plt.xlim([-7, 1])
plt.ylim([-3, 3])

plt.savefig('root_locus_diagram.png')
plt.close()
