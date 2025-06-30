#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as ds
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib
matplotlib.use("Agg",force=True)
from matplotlib import pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from plot_utils import SkewXAxes, ReadAIDA, Rect2Curv
from matplotlib.projections import register_projection
from matplotlib.ticker import (MultipleLocator, NullFormatter,
                               ScalarFormatter)

# Register the skew-T projection with matplotlib to plot with it.
register_projection(SkewXAxes)

print("Open GFS inputs (regridded and original)")
out = ds("data/debug/out_gfs.tile1.nc", "r")
in_gfs = ds("data/gfs_data_in/gfs_data.tile1.nc", "r")
print("Read contents")
gfs_lat = np.array(out.variables["geolat"])
gfs_lon = np.array(out.variables["geolon"])
#AIDA pressure level temperature regridded to GFS grid
regrid_250_t = np.array(out.variables["t_r"][0,:,:])
regrid_350_t = np.array(out.variables["t_r"][1,:,:])
regrid_t = np.array(out.variables["t_r"])
#AIDA-derived input
final_t = np.array(out.variables["t"])
ps = out.variables["ps"]
delp = out.variables["delp"]
#Original input
orig_t = np.array(in_gfs.variables["t"])

print("Read raw AIDA data")

aida_data = ReadAIDA("data/gfs_data_in/exp004murz_pernak_predict_PREP_test_for_GFS.nc")

#GFS lat/lon indexes of interest
x_ndx = 0
y_ndx = 129

p_int = np.zeros([delp.shape[0]+1,delp.shape[1],delp.shape[2]])
p = np.zeros(delp.shape)
#Construct GFS pressure
for k in range(1,p_int.shape[0]):
   p_int[k,:,:] = p_int[k-1,:,:] + delp[k-1,:,:]
   p[k-1,:,:] = (p_int[k-1,:,:] + p_int[k,:,:]) * 0.5

in_gfs.close()
out.close()
#Find the 350mb indexes
p_t = p - 35000
p_t = np.where(p_t < 0.0, 999999999.0, p)
ndx_350 = np.argmin(p_t, axis=0)

#Interpolate for temperature based on ndx_350
print("Interpolate for temperature")
gfs_int_350_t = np.zeros(p.shape[1:])
a = np.where(p > 0.0, p, -999)
subset = np.where(p > 0.0)
logp = np.zeros(p.shape) - 999.0
logp[subset] = np.log(p[subset])
log_350 = np.log(35000.0)
for i in range(p.shape[1]):
   for j in range(p.shape[2]):
      numer = (final_t[ndx_350[i,j],i,j] - final_t[ndx_350[i,j]-1,i,j]) * (log_350-logp[ndx_350[i,j]-1,i,j])
      denom = logp[ndx_350[i,j],i,j] - logp[ndx_350[i,j]-1,i,j]
      gfs_int_350_t[i,j] = final_t[ndx_350[i,j]-1,i,j] + numer / denom

print("Done interpolating")

diff_350_t = gfs_int_350_t - regrid_350_t
max_ndx = np.unravel_index(np.argmax(diff_350_t), diff_350_t.shape)
p_aida = [25000, 35000, 50000, 75000]

print("Plot raw AIDA 350mb temperature")
ai_350_t = aida_data["t"][1,:,:]
ai_lat = aida_data["geolat"]
ai_lon = aida_data["geolon"]
#convert lat/lon from 1d to 2d
#(ai_lat, ai_lon) = Rect2Curv(ai_lat, ai_lon)
map = plt.axes(projection=ccrs.PlateCarree())
map.set_global()
# # draw coastlines, country boundaries, fill continents.
map.coastlines(linewidth=0.25)
# # contour data over the map.
levels = np.linspace(200, 260, 31)
cs = map.contourf(ai_lon,ai_lat,ai_350_t,levels=levels,transform=ccrs.PlateCarree())
plt.colorbar(cs, fraction=0.026, pad=0.04)
plt.title('350mb Temperature (Raw AI-DA)')
plt.savefig("aida_350t_raw.png")
plt.close('all')

print("Plot regridded 350mb temperature")
map = plt.axes(projection=ccrs.PlateCarree())
map.set_global()
# # draw coastlines, country boundaries, fill continents.
map.coastlines(linewidth=0.25)
# # contour data over the map.
gfs_lon = np.where(gfs_lon > 180.0, gfs_lon - 360.0, gfs_lon)
levels = np.linspace(220, 260, 21)
cs = map.contourf(gfs_lon,gfs_lat,regrid_350_t,levels=levels,transform=ccrs.PlateCarree())
plt.colorbar(cs, fraction=0.026, pad=0.04)
plt.title('350mb Temperature (Regridded AI-DA)')
plt.savefig("regrid_350t.png")
plt.close('all')

print("plot 350mb interpolated temperatures")
map_350 = plt.axes(projection=ccrs.PlateCarree())
map_350.set_global()
# # draw coastlines, country boundaries, fill continents.
map_350.coastlines(linewidth=0.25)
# # contour data over the map.
levels = np.linspace(220, 260, 21)
cs2 = map_350.contourf(gfs_lon,gfs_lat,gfs_int_350_t,levels=levels,transform=ccrs.PlateCarree())
plt.colorbar(cs2, fraction=0.026, pad=0.04)
plt.title('350mb Temperature (GFS Orig Input)')
plt.savefig("orig_int_350t.png")
plt.close('all')

diff_350_t = gfs_int_350_t - regrid_350_t
map_diff = plt.axes(projection=ccrs.PlateCarree())
map_diff.set_global()
# # draw coastlines, country boundaries, fill continents.
map_diff.coastlines(linewidth=0.25)
# # contour data over the map.
levels = np.linspace(-.5, .5, 21)
cs3 = map_diff.contourf(gfs_lon,gfs_lat,diff_350_t,levels=levels,transform=ccrs.PlateCarree())
plt.colorbar(cs3, fraction=0.026, pad=0.04)
plt.title('350mb Temperature Difference')
plt.savefig("diff_350_t.png")
plt.close('all')

print("Plot skew-t")
####################################################
#Now plot skew-t log(p) plots

fig = plt.figure(figsize=(6.5875, 6.2125))
ax = fig.add_subplot(111,projection='skewx')

plt.grid(True)
ax.semilogy(final_t[:,0,129], p[:,0,129]/100.0, color='C2', label = 'Final AI-DA GFS Input')
ax.semilogy(regrid_t[:,0,129], [n/100.0 for n in p_aida], linestyle = '--', marker='o', color='C3', label = 'AI-DA In GFS Grid')
ax.semilogy(orig_t[:,0,129], p[:,0,129]/100.0, color='C4', linestyle = '', marker='.', label = "Orig GFS Input")

# Disables the log-formatting that comes with semilogy
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(np.linspace(100, 1000, 10))
ax.set_ylim(1030, 100)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_xlim(230,285)
ax.legend()

plt.title('Temperature (K)')
plt.savefig("skew_t.png")
plt.close('all')
