import netCDF4 as nc4
import numpy as np
tile = "2"
var = "zh"
ext = ""
var = var+ext
f1 = nc4.Dataset("out/gfs_data.tile"+tile+".nc")
f2 = nc4.Dataset("out.bak/gfs_data.tile"+tile+".nc")

v1=np.array(f1.variables[var])
v2=np.array(f2.variables[var])
d = v1 - v2

if(d.max() == 0 and d.min() == 0):
   print("variable is the same")
   exit()

geolon = np.array(f1.variables["geolon"+ext])
geolat = np.array(f1.variables["geolat"+ext])

for i in range(d.shape[0]):
   if(d[i,:,:].max() < 1.0 and d[i,:,:].min() > -1.0):
      continue
   for j in range(d.shape[1]):
      if(d[i,j,:].max() < 1.0 and d[i,j,:].min() > -1.0):
         continue
      for k in range(d.shape[2]):
         if(d[i,j,k] > 1.0 or d[i,j,k] < -1.0):
            print(i,j,k,d[i,j,k], geolon[j,k], geolat[j,k])
