#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as ds
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib import pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

out = ds("out_gfs.nc", "r")
gfs_lat = np.array(out.variables["geolat"])
gfs_lon = np.array(out.variables["geolon"])
interp_750_t = np.array(out.variables["t_aida"][0,:,:])

aida = ds("exp004murz_pernak_predict_PREP_test_for_GFS.nc", "r")

aida_out = aida.variables["tb_target"]
aida_750_t = np.zeros([aida.dimensions['nx'].size, aida.dimensions['ny'].size])
aida_750_t[:,:] = aida_out[0,:,:,0]

n_lat = aida.dimensions['nx'].size
n_lon = aida.dimensions['ny'].size
#Make a grid of longitude and latitude
aida_lon = np.linspace(0.0,360.0,n_lon)
aida_lat = np.linspace(-90,90,n_lat)

gfs_750_t = np.zeros([out.dimensions['lat'].size,out.dimensions['lon'].size])
gfs_750_t[:,:] = out.variables["t_aida"][0,:,:]

# set up orthographic map projection with
# # perspective of satellite looking down at 50N, 100W.
# # use low resolution coastlines.

map = plt.axes(projection=ccrs.PlateCarree())
map.set_global()
# # draw coastlines, country boundaries, fill continents.
map.coastlines(linewidth=0.25)
# # contour data over the map.
gfs_lon = np.where(gfs_lon > 180.0, gfs_lon - 360.0, gfs_lon)
cs = map.contourf(gfs_lon,gfs_lat,interp_750_t,20,linewidths=1.5,transform=ccrs.PlateCarree())
#cs = map.contour(aida_lon,aida_lat,aida_750_t,20,linewidths=1.5,transform=ccrs.PlateCarree())
g1 = map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
g1.labels_top=False
g1.labels_bottom=True
g1.labels_left=False
g1.labels_right=True
g1.xlines=False
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER
plt.colorbar(cs)
plt.title('750mb Temperature')
plt.savefig("contour.png")
