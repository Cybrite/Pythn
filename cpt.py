import numpy as np
import matplotlib.pyplot as plt
t = np.array([0, 5, 10, 15, 20, 25, 30, 35])
Ct = np.array([0, 0, 0, 5, 10, 5, 0, 0])

# Plotting Ct curve
plt.figure(figsize=(8, 5))
plt.plot(t, Ct, marker='o', linestyle='-', color='black', label='C(t)')
plt.title(f'Tracer Concentration vs Time\nArea under curve = {np.trapz(Ct, t)}')
plt.xlabel('Time t (s)')
plt.ylabel('Concentration C(t) (mg/dm³)')
plt.grid(True)
plt.show()

Et = Ct/np.trapz(Ct, t)
# Plotting Et curve
plt.figure(figsize=(8, 5))
plt.plot(t, Et, marker='o', linestyle='-', color='black', label='E(t)')
plt.title('Normalized Curve')
plt.xlabel('Time t (s)')
plt.ylabel('E(t)')
plt.grid(True)
plt.show()

tct = t * Ct
# Plotting t*Ct curve
plt.figure(figsize=(8, 5))
plt.plot(t, tct, marker='o', linestyle='-', color='black', label='tC(t)')
plt.title(f'tC(t) vs t curve\n Area under curve = {np.trapz(tct,t)}')
plt.xlabel('Time t (s)')
plt.ylabel('tC(t)')
plt.grid(True)
plt.show()

V = np.array([0, 0, 0, 1.25, 0, 1.25, 0, 0])
t = np.array([0, 5, 10, 15, 20, 25, 30, 35])

plt.figure(figsize=(8, 5))
plt.plot(t, V, marker='o', linestyle='-', color='black')
plt.title(f'Area under curve = {np.trapz(V,t)}')
plt.xlabel('Time t (s)')
plt.ylabel('((t-tm)^2) * E(t)')
plt.grid(True)
plt.show()