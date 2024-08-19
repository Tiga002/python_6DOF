import math
import numpy as np 

# Example usage:
def flat_earth_eom(t, x, amod):
  """FUNCTION flat_earth_eom.py contains the essential elements of a six-degree of freedom 
  simulation. The purpose of this function is to allow the numerical approximation of 
  solutions of the governing equations for an aircraft.
  
  The naming convention is <variable name>_<coordinate system if applicable>_<units>. For 
  example, the pitch rate, q, resolved in the body fixed frame, bf, with units of radians
  per second is named, q_b_rps.

  Arguments: 
    t - time [s], scalar
    x - state vector at time t [various units], numpy array
      x[0]  = u_b_mps, axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS
      x[1]  = v_b_mps, lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS
      x[2]  = w_b_mps, vertical velocity of CM wrt inertial CS resolved in aircraft body fixed CS
      x[3]  = p_b_rps, roll angular velocity of body fixed CS with respect to inertial CS
      x[4]  = q_b_rps, pitch angular velocity of body fixed CS with respect to inertial CS
      x[5]  = r_b_rps, yaw angular velocity of body fixed CS with respect to inertial CS
      x[6]  = phi_rad, roll angle
      x[7]  = theta_rad, pitch angle
      x[8]  = psi_rad, yaw angle
      x[9]  = p1_n_m, x-axis position of aircraft resolved in NED CS
      x[10] = p2_n_m, y-axis position of aircraft resolved in NED CS
      x[11] = p3_n_m, z-axis position of aircraft resolved in NED CS
    amod  = aircraft model data stored as a dictionary containing various parameters

  Returns:
    dx - the time derivative of each state in x (RHS of governing equations)

  History:
    Written by Ben Dickinson
      - Six degree of freedom equations written March 2024
  """
  
  # Preallocate left hand side of equations
  dx = np.empty((12,),dtype=float)

  # Assign current state values to variable names
  u_b_mps  = x[0]
  v_b_mps  = x[1]
  w_b_mps  = x[2]
  p_b_rps  = x[3]
  q_b_rps  = x[4]
  r_b_rps  = x[5]
  phi_rad   = x[6]
  theta_rad = x[7]
  psi_rad   = x[8]
  p1_n_m    = x[9]
  p2_n_m    = x[10]
  p3_n_m    = x[11]

  # Get mass and moments of inertia
  m_kg = amod["m_kg"]
  Jxz_b_kgm2 = amod["Jxz_b_kgm2"]
  Jxx_b_kgm2 = amod["Jxx_b_kgm2"]
  Jyy_b_kgm2 = amod["Jyy_b_kgm2"]
  Jzz_b_kgm2 = amod["Jzz_b_kgm2"]

  # Air data calculation (Mach, altitude, AoA, AoS) (Coming soon)
  
  # Atmosphere model (Coming soon)

  # Gravity acts normal to earth tangent CS
  gz_n_mps2 = 9.81
  
  # Resolve gravity in body coordinate system
  gx_b_mps2 = -math.sin(theta_rad) * gz_n_mps2
  gy_b_mps2 =  math.sin(phi_rad) * math.cos(theta_rad) * gz_n_mps2
  gz_b_mps2 =  math.cos(phi_rad) * math.cos(theta_rad) * gz_n_mps2

  # External forces (Coming soon)
  Fx_b_kgmps2 = 0
  Fy_b_kgmps2 = 0
  Fz_b_kgmps2 = 0
  
  # External moments (Coming soon)
  l_b_kgm2ps2 = 0
  m_b_kgm2ps2 = 0
  n_b_kgm2ps2 = 0
  
  # Denominator in roll and yaw rate equations
  Den = Jxx_b_kgm2 * Jzz_b_kgm2 - Jxz_b_kgm2**2
  
  # x-axis (roll-axis) velocity equation
  #  State: u_b_mps
  dx[0] = 1 / m_kg * Fx_b_kgmps2 + gx_b_mps2 - w_b_mps * q_b_rps + v_b_mps * r_b_rps
  
  # y-axis (pitch-axis) velocity equation
  #  State: v_b_mps
  dx[1] = 1 / m_kg*Fy_b_kgmps2 + gy_b_mps2 - u_b_mps*r_b_rps + w_b_mps*p_b_rps
  
  # z-axis (yaw-axis) velocity equation
  #  State: w_b_mps
  dx[2] = 1 / m_kg * Fz_b_kgmps2 + gz_b_mps2 - v_b_mps * p_b_rps + u_b_mps * q_b_rps
  
  # Roll equation
  #  State: p_b_rps
  dx[3] = (Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2  + Jzz_b_kgm2)    * p_b_rps * q_b_rps - \
          (Jzz_b_kgm2 * (Jzz_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2**2) * q_b_rps * r_b_rps +  \
           Jzz_b_kgm2 * l_b_kgm2ps2 + \
           Jxz_b_kgm2 * n_b_kgm2ps2)/Den
            
  # Pitch equation
  #  State: q_b_rps
  dx[4] = ((Jzz_b_kgm2 - Jxx_b_kgm2) * p_b_rps * r_b_rps - \
           Jxz_b_kgm2 * (p_b_rps**2 - r_b_rps**2) + m_b_kgm2ps2)/Jyy_b_kgm2
  
  # Yaw equation
  #  State: r_b_rps
  dx[5] = ((Jxx_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2**2) * p_b_rps * q_b_rps + \
            Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2)     * q_b_rps * r_b_rps + \
            Jxz_b_kgm2 * l_b_kgm2ps2 + \
            Jxz_b_kgm2 * n_b_kgm2ps2)/Den
  
  # Kinematic equations
  dx[6] = p_b_rps + math.sin(phi_rad)*math.tan(theta_rad)*q_b_rps + \
                    math.cos(phi_rad)*math.tan(theta_rad)*r_b_rps
  
  dx[7] = math.cos(phi_rad)*q_b_rps - \
          math.sin(phi_rad)*r_b_rps

  dx[8] = math.sin(phi_rad)/math.cos(theta_rad)*q_b_rps + \
          math.cos(phi_rad)/math.cos(theta_rad)*r_b_rps

  # Position (Navigation) equations (Coming soon)
  dx[9]  = 0
  dx[10] = 0
  dx[11] = 0

  return dx