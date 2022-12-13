import netCDF4 as nc4
import numpy as np

f = nc4.Dataset("gfs_data.tile6.nc","r")

for variable in f.variables:
   print(variable)
   a = np.array(f.variables[variable])
   print(a.min(), a.max())
