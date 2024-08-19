import numpy as np  

def forward_euler(f, t_s, x, h_s, vmod, amod):
  """
  Performs forward Euler integration to approximate the solution of a differential equation.

  Input Args:
      f:   A function representing the right-hand side of the differential equation (dx/dt = f(t, x)).
      t_s: A vector of points in time at which numerical solutions will be approximated
      x:   The numerically approximated solution data to the DE, f
      h_s: The step size in seconds

  Returns:
      t_s: A vector of points in time at which numerical solutions was approximated
      x:   The numerically approximated solution data to the DE, f
      
  """

  for i in range(1, len(t_s)):
    x[:,i] = x[:,i-1] + h_s * f(t_s[i-1], x[:,i-1], vmod, amod)

  return t_s, x
