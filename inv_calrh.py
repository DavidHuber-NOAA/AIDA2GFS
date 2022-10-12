
def calc_q(p,t,rh):
    """
Author: David Huber

Calculates specific humidy at given pressures, temperatures, and relative humidities
Inputs should be array-like
The output will be a numpy array

In:
    p (Pressure in Pascals, 3d)
    t (Temperature in Kelvin, 3d)
    rh (Relative humidity as a decimal, 3d)

Out:
    q (Specific humidity in kg/kg, 3d)
    """

    import numpy as np
    from math import e
    #Define constants
    p_q0 = 379.90516 # Pascals; Reference vapor pressure?
    a2 = 17.2693882 # Unitless; ?
    a3 = 273.16 # K; Triple point temperature
    a4 = 35.86 # K; Reference temperature?
    rh_min=1.0E-6 # Unitless; Minimum relative humidity

    #Get dimensions of the inputs and make sure they match
    dims = np.shape(p)
    if(dims != np.shape(t) or dims != np.shape(rh)):
        raise ValueError("Dimension size mismatch in p, t, and/or rh")

    #q_c is saturation specific humidity
    q_c = p_q0/p * e ** (a2 * (t - a3) / (t - a4))

    #Establish min/max rh
    rh_t = rh
    rh_t = np.where(rh_t > 1.0, 1.0, np.where(rh_t < rh_min, rh_min, rh_t))

    #Calculate specific humidity
    q = q_c * rh_t
    return q
