import matplotlib.pyplot as plt
import numpy as np  
import math

from numerical_integrators import numerical_integration_methods
from governing_equations import flat_earth_eom

#==============================================================================
# Part 1: Initialization of simulation
#==============================================================================

# Define vehicle (a sphere for now)
r_sphere_m = 0.08
m_sphere_kg = 5
J_sphere_kgm2 = 0.4*m_sphere_kg*r_sphere_m**2

amod = { "m_kg" : 1, \
        "Jxz_b_kgm2" : 0, \
        "Jxx_b_kgm2" : J_sphere_kgm2, \
        "Jyy_b_kgm2" : J_sphere_kgm2, \
        "Jzz_b_kgm2" : J_sphere_kgm2 } # Aircraft model properties

# Set initial conditions (these conditions may be loaded from an aircraft
# trim routine in future versions of the code)
u0_bf_mps  = 0
v0_bf_mps  = 0
w0_bf_mps  = 0
p0_bf_rps  = 0
q0_bf_rps  = 0
r0_bf_rps  = 0
phi0_rad   = 0*math.pi/180
theta0_rad = 90*math.pi/180
psi0_rad   = 0
p10_n_m    = 0
p20_n_m    = 0
p30_n_m    = 0

x0 = np.array([
    u0_bf_mps,  # x-axis body-fixed velocity (m/s)
    v0_bf_mps,  # y-axis body-fixed velocity (m/s)
    w0_bf_mps,  # z-axis body-fixed velocity (m/s)
    p0_bf_rps,  # roll rate (rad/s)
    q0_bf_rps,  # pitch rate (rad/s)
    r0_bf_rps,  # yaw rate (rad/s)
    phi0_rad,   # roll angle (rad)
    theta0_rad, # pitch angle (rad)
    psi0_rad,   # yaw angle (rad)
    p10_n_m,    # x-axis position (N*m)
    p20_n_m,    # y-axis position (N*m)
    p30_n_m,    # z-axis position (N*m)
])

# Get number of elements in x0
nx0 = x0.size

# Set time conditions
t0_s = 0.0
tf_s = 10.0
h_s  = 0.005

#==============================================================================
# Part 2: Numerically approximate solutions to the governing equations
#==============================================================================

# Preallocate the solution array
t_s = np.arange( t0_s, tf_s + h_s, h_s ); nt_s = t_s.size
x   = np.zeros((nx0, nt_s))

# Assign the initial condition, x0, to solution array, x
x[:, 0] = x0 

# Perform forward Euler integration
t_s, x = numerical_integration_methods.forward_euler(flat_earth_eom.flat_earth_eom, t_s, x, h_s, amod)

# Data post-processing actions (Coming soon)

#==============================================================================
# Part 3: Plot data in figures
#==============================================================================

# Create subplots and set layout
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
fig.set_facecolor('black')  

# Axial velocity u^b_CM/n
axes[0, 0].plot(t_s, x[0,:], color='yellow')
axes[0, 0].set_xlabel('Time [s]', color='white')
axes[0, 0].set_ylabel('u [m/s]', color='white')
axes[0, 0].grid(True)
axes[0, 0].set_facecolor('black')
axes[0, 0].tick_params(colors = 'white')

# y-axis velocity v^b_CM/n
axes[0, 1].plot(t_s, x[1,:], color='yellow')
axes[0, 1].set_xlabel('Time [s]', color='white')
axes[0, 1].set_ylabel('v [m/s]', color='white')
axes[0, 1].grid(True)
axes[0, 1].set_facecolor('black')
axes[0, 1].tick_params(colors = 'white')

# z-axis velocity w^b_CM/n
axes[0, 2].plot(t_s, x[2,:], color='yellow')
axes[0, 2].set_xlabel('Time [s]', color='white')
axes[0, 2].set_ylabel('w [m/s]', color='white')
axes[0, 2].grid(True)
axes[0, 2].set_facecolor('black')
axes[0, 2].tick_params(colors = 'white')

# Roll angle, phi
axes[0, 3].plot(t_s, x[6,:], color='yellow')
axes[0, 3].set_xlabel('Time [s]', color='white')
axes[0, 3].set_ylabel('phi [rad]', color='white')
axes[0, 3].grid(True)
axes[0, 3].set_facecolor('black')
axes[0, 3].tick_params(colors = 'white')

# Roll rate p^b_b/n
axes[1, 0].plot(t_s, x[3,:], color='yellow')
axes[1, 0].set_xlabel('Time [s]', color='white')
axes[1, 0].set_ylabel('p [r/s]', color='white')
axes[1, 0].grid(True)
axes[1, 0].set_facecolor('black')
axes[1, 0].tick_params(colors = 'white')

# Pitch rate q^b_b/n
axes[1, 1].plot(t_s, x[4,:], color='yellow')
axes[1, 1].set_xlabel('Time [s]', color='white')
axes[1, 1].set_ylabel('q [r/s]', color='white')
axes[1, 1].grid(True)
axes[1, 1].set_facecolor('black')
axes[1, 1].tick_params(colors = 'white')

# Yaw rate r^b_b/n
axes[1, 2].plot(t_s, x[5,:], color='yellow')
axes[1, 2].set_xlabel('Time [s]', color='white')
axes[1, 2].set_ylabel('r [r/s]', color='white')
axes[1, 2].grid(True)
axes[1, 2].set_facecolor('black')
axes[1, 2].tick_params(colors = 'white')

# Pitch angle, theta
axes[1, 3].plot(t_s, x[7,:], color='yellow')
axes[1, 3].set_xlabel('Time [s]', color='white')
axes[1, 3].set_ylabel('theta [rad]', color='white')
axes[1, 3].grid(True)
axes[1, 3].set_facecolor('black')
axes[1, 3].tick_params(colors = 'white')

plt.tight_layout()
plt.savefig('saved_figures/sphere_drop_test_3.png')
plt.show()
