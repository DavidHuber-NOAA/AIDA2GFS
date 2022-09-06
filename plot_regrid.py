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
#AIDA pressure level temperature regridded to GFS grid
regrid_250_t = np.array(out.variables["t_r"][0,:,:])
regrid_350_t = np.array(out.variables["t_r"][1,:,:])
regrid_t = np.array(out.variables["t_r"])
#AIDA-derived input
gfs_t = np.array(out.variables["t"])

ps = out.variables["ps"]
delp = out.variables["delp"]
p = np.zeros(delp.shape)
#Construct GFS pressure
p[0,:,:] = ps[:,:] - delp[0,:,:]
for lev in range(1,delp.shape[0]-1):
   p[lev,:,:] = p[lev-1,:,:] - delp[lev,:,:]

p[-1,:,:] = 0.0

#Find the 350mb indexes
p_t = p - 35000
p_t = np.where(p_t < 0.0, -999999999.0, p)
ndx_350 = np.argmin(p_t, axis=0)

#Interpolate for temperature based on ndx_350
gfs_int_350_t = np.zeros(p.shape[1:])
a = np.where(p > 0.0, p, -999)
subset = np.where(p > 0.0)
logp = np.zeros(p.shape) - 999.0
logp[subset] = np.log(p[subset])
log_350 = np.log(35000.0)
for i in range(p.shape[1]):
   for j in range(p.shape[2]):
      numer = (gfs_t[ndx_350[i,j],i,j] - gfs_t[ndx_350[i,j]-1,i,j]) * (log_350-logp[ndx_350[i,j]-1,i,j])
      denom = logp[ndx_350[i,j],i,j] - logp[ndx_350[i,j]-1,i,j]
      gfs_int_350_t[i,j] = gfs_t[ndx_350[i,j]-1,i,j] + numer / denom

diff_350_t = gfs_int_350_t - regrid_350_t
max_ndx = np.unravel_index(np.argmax(diff_350_t), diff_350_t.shape)
p_aida = [25000, 35000, 50000, 75000]

# set up orthographic map projection with
# # perspective of satellite looking down at 50N, 100W.
# # use low resolution coastlines.

map = plt.axes(projection=ccrs.PlateCarree())
map.set_global()
# # draw coastlines, country boundaries, fill continents.
map.coastlines(linewidth=0.25)
# # contour data over the map.
gfs_lon = np.where(gfs_lon > 180.0, gfs_lon - 360.0, gfs_lon)
levels = np.linspace(200, 240, 21)
cs = map.contourf(gfs_lon,gfs_lat,regrid_250_t,levels=levels,transform=ccrs.PlateCarree())
#g1 = map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
#g1 = map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
#g1.labels_top=False
#g1.labels_bottom=True
#g1.labels_left=False
#g1.labels_right=True
#g1.xlines=False
#g1.xformatter = LONGITUDE_FORMATTER
#g1.yformatter = LATITUDE_FORMATTER
plt.colorbar(cs, fraction=0.046, pad=0.04)
plt.title('250mb Temperature')
plt.savefig("regrid_250t.png")
plt.close('all')

map_350 = plt.axes(projection=ccrs.PlateCarree())
map_350.set_global()
# # draw coastlines, country boundaries, fill continents.
map_350.coastlines(linewidth=0.25)
# # contour data over the map.
levels = np.linspace(230, 270, 21)
cs2 = map_350.contourf(gfs_lon,gfs_lat,gfs_int_350_t,levels=levels,transform=ccrs.PlateCarree())
plt.colorbar(cs2, fraction=0.026, pad=0.04)
plt.title('350mb Temperature')
plt.savefig("regrid_350t.png")
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












############################################################
#skew-t log(p) classes and functions
# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
from contextlib import ExitStack

from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.projections import register_projection
from matplotlib.ticker import (MultipleLocator, NullFormatter,
                               ScalarFormatter)

class SkewXTick(maxis.XTick):
    def draw(self, renderer):
        # When adding the callbacks with `stack.callback`, we fetch the current
        # visibility state of the artist with `get_visible`; the ExitStack will
        # restore these states (`set_visible`) at the end of the block (after
        # the draw).
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line,
                           self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())
            needs_lower = transforms.interval_contains(
                self.axes.lower_xlim, self.get_loc())
            needs_upper = transforms.interval_contains(
                self.axes.upper_xlim, self.get_loc())
            self.tick1line.set_visible(
                self.tick1line.get_visible() and needs_lower)
            self.label1.set_visible(
                self.label1.get_visible() and needs_lower)
            self.tick2line.set_visible(
                self.tick2line.get_visible() and needs_upper)
            self.label2.set_visible(
                self.label2.get_visible() and needs_upper)
            super(SkewXTick, self).draw(renderer)

    def get_view_interval(self):
        return self.axes.xaxis.get_view_interval()


# This class exists to provide two separate sets of intervals to the tick,
# as well as create instances of the custom tick
class SkewXAxis(maxis.XAxis):
    def _get_tick(self, major):
        return SkewXTick(self.axes, None, '', major=major)

    def get_view_interval(self):
        return self.axes.upper_xlim[0], self.axes.lower_xlim[1]


# This class exists to calculate the separate data range of the
# upper X-axis and draw the spine there. It also provides this range
# to the X-axis artist for ticking and gridlines
class SkewSpine(mspines.Spine):
    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim


# This class handles registration of the skew-xaxes as a projection as well
# as setting up the appropriate transformations. It also overrides standard
# spines and axes instances as appropriate.
class SkewXAxes(Axes):
    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='skewx')``.
    name = 'skewx'

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines['top'].register_axis(self.xaxis)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)

    def _gen_axes_spines(self):
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        rot = 30

        # Get the standard transform setup from the Axes base class
        super()._set_lim_and_transforms()

        # Need to put the skew in the middle, after the scale and limits,
        # but before the transAxes. This way, the skew is done in Axes
        # coordinates thus performing the transform around the proper origin
        # We keep the pre-transAxes transform around for other users, like the
        # spines for finding bounds
        self.transDataToAxes = (
            self.transScale
            + self.transLimits
            + transforms.Affine2D().skew_deg(rot, 0)
        )
        # Create the full transform from Data to Pixels
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform = (
            transforms.blended_transform_factory(
                self.transScale + self.transLimits,
                transforms.IdentityTransform())
            + transforms.Affine2D().skew_deg(rot, 0)
            + self.transAxes
        )

    @property
    def lower_xlim(self):
        return self.axes.viewLim.intervalx

    @property
    def upper_xlim(self):
        pts = [[0., 1.], [1., 1.]]
        return self.transDataToAxes.inverted().transform(pts)[:, 0]

# Now register the projection with matplotlib so the user can select it.
register_projection(SkewXAxes)






####################################################
#Now plot skew-t log(p) plots

fig = plt.figure(figsize=(6.5875, 6.2125))
ax = fig.add_subplot(111,projection='skewx')

plt.grid(True)
ax.semilogy(gfs_t[:,0,129], p[:,0,129]/100.0, color='C2', label = 'GFS Grid')
ax.semilogy(regrid_t[:,0,129], [n/100.0 for n in p_aida], linestyle = '--', marker='o', color='C3', label = 'AI-DA Grid')

# Disables the log-formatting that comes with semilogy
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(np.linspace(100, 1000, 10))
ax.set_ylim(760, 240)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_xlim(230,265)
ax.legend()

plt.title('Vertically interpolated temperature')
plt.savefig("skew_t.png")
plt.close('all')
