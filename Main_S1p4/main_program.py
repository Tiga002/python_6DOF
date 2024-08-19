import matplotlib.pyplot as plt
import numpy as np  
import math
import ussa1976

from numerical_integrators import numerical_integration_methods
from governing_equations import flat_earth_eom
from tools.Interpolators import fastInterp1

#==============================================================================
# Part 1: Initialization of simulation
#==============================================================================

# A. Atmospheric data
atmosphere = ussa1976.compute()

# Get essential gravity and atmospheric data
alt_m     = atmosphere["z"].values
rho_kgpm3 = atmosphere["rho"].values
c_mps     = atmosphere["cs"].values
g_mps2    = ussa1976.core.compute_gravity(alt_m)

amod = { "alt_m"    : alt_m, \
        "rho_kgpm3" : rho_kgpm3, \
        "c_mps"     : c_mps, \
        "g_mps2"    : g_mps2 } # Atmosphere and gravity data

# B. Define vehicle 

# Material densities
rho_lead_kgpm3      = 11300
rho_cast_iron_kgpm3 = 7000

# 50 caliber lead ball
#r_sphere_m  = 0.495*(2.54/12)/10
#m_sphere_kg = 4/3*rho_lead_kgpm3*math.pi*r_sphere_m**3
#vehicle_name = "50 Cal Lead Round Shot"

#  
# 12 pounder Carronade (cast iron)
r_sphere_m = 4.40*(2.54/12)/10
m_sphere_kg = 4/3*rho_cast_iron_kgpm3*math.pi*r_sphere_m**3
vehicle_name = "Carronade 12 lb (5.4 kg) Cannonball"

J_sphere_kgm2   = 0.4*m_sphere_kg*r_sphere_m**2
CD_approx       = 0.5
Aref_m2         = math.pi*r_sphere_m**2
Vterm_mps       = math.sqrt((2*m_sphere_kg*9.81)/(1.2*CD_approx*Aref_m2))

vmod = {"V_name"     : vehicle_name, \
        "m_kg"       : m_sphere_kg, \
        "Jxz_b_kgm2" : 0, \
        "Jxx_b_kgm2" : J_sphere_kgm2, \
        "Jyy_b_kgm2" : J_sphere_kgm2, \
        "Jzz_b_kgm2" : J_sphere_kgm2, \
        "r_sphere_m" : r_sphere_m, \
        "m_sphere_kg": m_sphere_kg, \
        "CD_approx"  : CD_approx, \
        "Aref_m2"    : Aref_m2, \
        "Vterm_mps"  : Vterm_mps } 

print(f"The analytical terminal velocity is {Vterm_mps:.2f} m/s.")

# Set initial conditions (these conditions may be loaded from an aircraft
# trim routine in future versions of the code)
u0_bf_mps  = 0.001 # avoids divide by zero
v0_bf_mps  = 0
w0_bf_mps  = 0
p0_bf_rps  = 0
q0_bf_rps  = 0
r0_bf_rps  = 0
phi0_rad   =   0*math.pi/180
theta0_rad = -90*math.pi/180
psi0_rad   = 0
p10_n_m    = 0
p20_n_m    = 0
p30_n_m    = -50000

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
tf_s = 200.0
h_s  = 0.01

#==============================================================================
# Part 2: Numerically approximate solutions to the governing equations
#==============================================================================

# Preallocate the solution array
t_s = np.arange( t0_s, tf_s + h_s, h_s ); nt_s = t_s.size
x   = np.zeros((nx0, nt_s))

# Assign the initial condition, x0, to solution array, x
x[:, 0] = x0 

# Perform forward Euler integration
t_s, x = numerical_integration_methods.forward_euler(flat_earth_eom.flat_earth_eom, \
    t_s, x, h_s, vmod, amod)

# B. Data post-processing 

# Airspeed
True_Airspeed_mps = np.zeros((nt_s,1))
for i, element in enumerate(t_s):
    True_Airspeed_mps[i,0] = math.sqrt(x[0,i]**2 + x[1,i]**2 + x[2,i]**2)
    
# Altitude, speed of sound, and air density
Altitude_m  = np.zeros((nt_s,1))
Cs_mps      = np.zeros((nt_s,1))
Rho_kgpm3   = np.zeros((nt_s,1))

for i, element in enumerate(t_s):
    Altitude_m[i,0] = -x[11,i]
    Cs_mps[i,0]     = fastInterp1(amod["alt_m"], amod["c_mps"],    Altitude_m[i,0])
    Rho_kgpm3[i,0]  = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], Altitude_m[i,0])
    
# Angle of attack
Alpha_rad = np.zeros((nt_s,1))
for i, element in enumerate(t_s):
        
    if x[0,i] == 0 and x[2,i] == 0:
        w_over_v = 0
    else:
        w_over_v = x[2,i]/x[0,i]
        
    Alpha_rad[i,0] = math.atan(w_over_v)
    
# Angle of side slip
Beta_rad = np.zeros((nt_s,1))
for i, element in enumerate(t_s):
        
    if x[1,i] == 0 and True_Airspeed_mps[i,0] == 0:
        v_over_VT = 0
    else:
        v_over_VT = x[1,i]/True_Airspeed_mps[i,0]
        
    Beta_rad[i,0] = math.atan(v_over_VT)
    
# Mach Number
Mach = np.zeros((nt_s,1))
for i, element in enumerate(t_s):
    Mach[i,0] = True_Airspeed_mps[i,0]/Cs_mps[i,0]
    
print(f"The numerical terminal velocity is {x[0,-1]:.2f} m/s.")

#==============================================================================
# Part 3: Plot data in figures
#==============================================================================

# Create figure of translational and rotation states
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
fig.set_facecolor('black')  
fig.suptitle(vmod["V_name"], fontsize=14, fontweight='bold', color='yellow')

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
#plt.savefig('saved_figures/sphere_drop_test_3.png')
plt.show(block=False)

# Create figure of position states
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.set_facecolor('black') 
fig.suptitle(vmod["V_name"], fontsize=14, fontweight='bold', color='cyan') 

# North position p1^n_CM/T
axes[0,0].plot(t_s, x[9,:], color='cyan')
axes[0,0].set_xlabel('Time [s]', color='white')
axes[0,0].set_ylabel('North [m]', color='white')
axes[0,0].grid(True)
axes[0,0].set_facecolor('black')
axes[0,0].tick_params(colors = 'white')

# East position p2^n_CM/T
axes[0,1].plot(t_s, x[10,:], color='cyan')
axes[0,1].set_xlabel('Time [s]', color='white')
axes[0,1].set_ylabel('East [m]', color='white')
axes[0,1].grid(True)
axes[0,1].set_facecolor('black')
axes[0,1].tick_params(colors = 'white')

# Altitude
axes[0,2].plot(t_s, -x[11,:], color='cyan')
axes[0,2].set_xlabel('Time [s]', color='white')
axes[0,2].set_ylabel('Altitude [m]', color='white')
axes[0,2].grid(True)
axes[0,2].set_facecolor('black')
axes[0,2].tick_params(colors = 'white')

# North vs East position p2^n_CM/T
axes[1,0].plot(x[10,:], x[9,:], color='cyan')
axes[1,0].set_xlabel('East [s]', color='white')
axes[1,0].set_ylabel('North [m]', color='white')
axes[1,0].grid(True)
axes[1,0].set_facecolor('black')
axes[1,0].tick_params(colors = 'white')

# Altitude vs East position p2^n_CM/T
axes[1,1].plot(x[10,:], -x[11,:], color='cyan')
axes[1,1].set_xlabel('East [s]', color='white')
axes[1,1].set_ylabel('Altitude [m]', color='white')
axes[1,1].grid(True)
axes[1,1].set_facecolor('black')
axes[1,1].tick_params(colors = 'white')

# Altitude vs North
axes[1,2].plot(x[9,:], -x[11,:], color='cyan')
axes[1,2].set_xlabel('North [s]', color='white')
axes[1,2].set_ylabel('Altitude [m]', color='white')
axes[1,2].grid(True)
axes[1,2].set_facecolor('black')
axes[1,2].tick_params(colors = 'white')

plt.tight_layout()
#plt.savefig('saved_figures/positions_3.png')
plt.show(block=False)

# Create figure of air data
fig, axes = plt.subplots(1, 3, figsize=(10, 6))
fig.set_facecolor('black')  
fig.suptitle(vmod["V_name"], fontsize=14, fontweight='bold', color='magenta')

# Angle of attack
axes[0].plot(t_s, Alpha_rad*180/3.14, color='magenta')
axes[0].set_xlabel('Time [s]', color='white')
axes[0].set_ylabel('Angle of Attack [deg]', color='white')
axes[0].set_ylim(-30,30)
axes[0].grid(True)
axes[0].set_facecolor('black')
axes[0].tick_params(colors = 'white')

# Angle of side slip
axes[1].plot(t_s, Beta_rad*180/3.14, color='magenta')
axes[1].set_xlabel('Time [s]', color='white')
axes[1].set_ylabel('Angle of Side Slip [deg]', color='white')
axes[1].set_ylim(-30,30)
axes[1].grid(True)
axes[1].set_facecolor('black')
axes[1].tick_params(colors = 'white')

# Mach
axes[2].plot(t_s, Mach, color='magenta')
axes[2].set_xlabel('Time [s]', color='white')
axes[2].set_ylabel('Mach Number', color='white')
axes[2].grid(True)
axes[2].set_facecolor('black')
axes[2].tick_params(colors = 'white')

plt.tight_layout()
#plt.savefig('saved_figures/air_data_3.png')
plt.show()