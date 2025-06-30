###Define skew-t###
############################################################
#skew-t log(p) classes and functions
# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
from contextlib import ExitStack

from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import numpy as np

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

def ReadAIDA(filename):

   from netCDF4 import Dataset as ds
   convert_rh="no"

   with ds(filename,"r") as aida_fh:
      lev = 4

      #Read specific humidity, u, v, t, z, geolat, geolon
      #All are on isobaric levels (250, 350, 500, and 750mb)

      p = np.array([25000, 35000, 50000, 75000])

      # All quantities are stored in the tb_target array, which is read then parsed
      aida_out = aida_fh.variables["tb_target"]
      lon = aida_fh.dimensions['ny'].size + 1
      # Dim lon+1 so we can repeat 0 lon and 360 lon
      lat = aida_fh.dimensions['nx'].size

      #Make a grid of longitude and latitude
      geolon = np.linspace(0.0,360.0, lon)
      geolat = np.linspace(-90.0, 90.0, lat)

      # Define wind, temperature, specific humidity, and geopotential height
      u = np.zeros((lev,lat,lon))
      v = np.zeros((lev,lat,lon))
      t = np.zeros((lev,lat,lon))
      rh_or_q = np.zeros((lev,lat,lon))
      zh = np.zeros((lev,lat,lon))

      #Read in the arrays; aida_data goes from 90 to -90, 0 to 359.65 (or so)
      #Repeat the first longitudinal slice at the end of each array so the
      #data goes from 0 to 360.
      for i in range(lev):
         t[i,:,:-1] = aida_out[0,::-1,:,lev*0+i]
         t[i,:,-1] = t[i,:,0]

         rh_or_q[i,:,:-1] = aida_out[0,::-1,:,lev*1+i]
         rh_or_q[i,:,-1] = rh_or_q[i,:,0]

         zh[i,:,:-1] = aida_out[0,::-1,:,lev*2+i]
         zh[i,:,-1] = zh[i,:,0]

         u[i,:,:-1] = aida_out[0,::-1,:,lev*3+i]
         u[i,:,-1] = u[i,:,0]

         v[i,:,:-1] = aida_out[0,::-1,:,lev*4+i]
         v[i,:,-1] = v[i,:,0]

   #If the input water vapor quantity is RH, convert to q
   if(convert_rh.lower() == "yes"):
      #Convert RH to a decimal
      rh = rh_or_q / 100.0
      #Convert P to a 3-d field
      p3d = np.zeros((lev,lat,lon))
      for i in range(lev):
         p3d[i,:,:] = p[i]

      sphum = np.zeros(rh.shape)
      sphum = calc_q(p3d, t, rh)
   else:
      sphum = rh_or_q

   #Create a dict with all of the AIDA data in it
   aida_data = { 'lev' : lev,
            'lon' : lon,
            'lat' : lat,
            'geolat' : geolat,
            'geolon' : geolon,
            'p' : p,
            'u' : u,
            'v' : v,
            't' : t,
            'sphum' : sphum,
            'zh' : zh }

   return aida_data

def Rect2Curv(in_geolat, in_geolon):
   curv_geolat = np.zeros([in_geolat.shape[0],in_geolon.shape[0]])
   curv_geolon = np.zeros([in_geolat.shape[0],in_geolon.shape[0]])
   for i in range(in_geolat.shape[0]):
      for j in range(in_geolon.shape[0]):
         curv_geolat[i,j] = in_geolat[i]
         curv_geolon[i,j] = in_geolon[j]
   return(curv_geolat, curv_geolon)
