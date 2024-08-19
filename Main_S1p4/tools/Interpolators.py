import numpy as np

def fastInterp1(x, y, xi):
    """
    Performs linear interpolation for a given data set.

    Args:
        x: Array of independent data points.
        y: Array of dependent data points.
        xi: Scalar value to interpolate.

    Returns:
        yi: Interpolated dependent data value for xi.
    """

    # Ensure x is a column vector
    x = x.reshape(-1, 1) 

    # Check if lengths of x and y match
    if len(x) != len(y):
      raise ValueError("Lengths of x and y must be equal.")

    # Create intervals array
    iia_r1 = np.array( [ -np.inf,   x[0,0],    y[0],   y[0], 1 ] )
    iia_mr = np.array( [  x[:-1,0],   x[1:,0], y[:-1],  y[1:], 2 * np.ones(len(x) - 1) ] )
    iia_mr = iia_mr.transpose()
    iia_rn = np.array( [   x[-1,0], np.inf,  y[-1],  y[-1], 3 ] )
    iia = np.vstack([iia_r1, iia_mr, iia_rn])

    # Find the relevant interval for xi
    xyc = iia[ (xi > iia[:, 0]) & (xi <= iia[:, 1]) , :]

    # Extract data from the interval
    x0 = xyc[0,0] 
    x1 = xyc[0,1]
    y0 = xyc[0,2]
    y1 = xyc[0,3]
    ic = xyc[0,4]

    # Perform interpolation based on the code
    if ic == 2:
      yi = (y0 * (x1 - xi) + y1 * (xi - x0)) / (x1 - x0)
    elif ic == 1:
      yi = y0
    elif ic == 3:
      yi = y1
    else:
      raise RuntimeError("Interpolation failure.")

    return yi
