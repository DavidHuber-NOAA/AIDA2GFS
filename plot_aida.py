#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as ds
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib import pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

aida = ds("exp004murz_pernak_predict_PREP_test_for_GFS.nc", "r")

aida_out = aida.variables["tb_target"]
aida_750_t = np.zeros([aida.dimensions['nx'].size, aida.dimensions['ny'].size])
aida_750_t[:,:] = aida_out[0,::-1,:,0]

n_lat = aida.dimensions['nx'].size
n_lon = aida.dimensions['ny'].size
lon_step = 360.0 / float(n_lon-1)
lat_step = 180.0 / float(n_lat-1)
#Make a grid of longitude and latitude
lon = np.asarray([ float(i) * lon_step for i in range(n_lon)])
lat = np.asarray([ float(i) * lat_step - 90 for i in range(n_lat)])

map = plt.axes(projection=ccrs.PlateCarree())
map.set_global()
# # draw coastlines, country boundaries, fill continents.
map.coastlines(linewidth=0.25)
cs = map.contourf(lon,lat,aida_750_t,20,linewidths=1.5,transform=ccrs.PlateCarree())
#cs = map.contour(lon,lat,aida_750_t,20,linewidths=1.5,transform=ccrs.PlateCarree())
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
plt.savefig("aida_750t.png")
