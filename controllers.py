# Controller Comparison: PI, PD, PID in Python
# Chemical Engineering Process: First Order System
# G(s) = 1/(10s + 1)

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

# --- Process Model (First-order plant) ---
G = ctrl.TransferFunction([1], [10, 1])

# --- Controller parameters ---
Kp = 2
Ki = 0.5
Kd = 1

# --- PI, PD, PID Controllers ---
PI  = ctrl.TransferFunction([Kp, Ki], [1, 0])
PD  = ctrl.TransferFunction([Kd, Kp], [1])
PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

# --- Closed Loop Responses ---
CL_PI  = ctrl.feedback(PI*G, 1)
CL_PD  = ctrl.feedback(PD*G, 1)
CL_PID = ctrl.feedback(PID*G, 1)

# --- Step Response Plot ---
t = np.linspace(0, 50, 500)

t1, y1 = ctrl.step_response(CL_PI, t)
t2, y2 = ctrl.step_response(CL_PD, t)
t3, y3 = ctrl.step_response(CL_PID, t)

plt.figure()
plt.plot(t1, y1, label='PI Controller')
plt.plot(t2, y2, label='PD Controller')
plt.plot(t3, y3, label='PID Controller')
plt.title("Step Response Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# --- Performance Metrics Function ---
def metrics(cl_sys, name):
    info = ctrl.step_info(cl_sys)
    print(f"\n=== {name} Controller Info ===")
    print(info)
    return info

PI_info  = metrics(CL_PI,  "PI")
PD_info  = metrics(CL_PD,  "PD")
PID_info = metrics(CL_PID, "PID")

# --- Bar Chart of Rise, Overshoot, Settling ---
values = np.array([
    [PI_info['RiseTime'],  PD_info['RiseTime'],  PID_info['RiseTime']],
    [PI_info['Overshoot'], PD_info['Overshoot'], PID_info['Overshoot']],
    [PI_info['SettlingTime'], PD_info['SettlingTime'], PID_info['SettlingTime']]
]).T

labels = ['PI', 'PD', 'PID']
metrics_names = ['Rise Time', 'Overshoot', 'Settling Time']

plt.figure()
bar_width = 0.25
x = np.arange(len(labels))

for i in range(3):
    plt.bar(x + i*bar_width, values[:, i], width=bar_width, label=metrics_names[i])

plt.xticks(x + bar_width, labels)
plt.ylabel("Performance Value")
plt.title("Controller Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()
