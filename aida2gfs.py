import netCDF4 as nc
from scipy.interpolate import interp2d
import numpy as np

def get_aida(aida_out_fname):

   aida_fh = nc.Dataset(aida_out_fname, 'r')
   n_lev = 4

   #Read specific humidity, u, v, t, z, lat, lon
   #All are on isobaric levels (750, 500, 350, and 250mb)

   p = np.array([75000, 50000, 35000, 25000])

   # All quantities are stored in the tb_target array, which is read then parsed
   aida_out = aida_fh.variables["tb_target"]
   n_lon = aida_fh.dimensions['ny'].size + 1
   # Dim n_lon+1 so we can repeat 0 lon and 360 lon
   n_lat = aida_fh.dimensions['nx'].size

   #Make a grid of longitude and latitude
   lon = np.linspace(0.0,360.0, n_lon)
   lat = np.linspace(-90.0, 90.0, n_lat)

   # Define wind, temperature, specific humidity, and geopotential height
   u = np.zeros((n_lev,n_lat,n_lon))
   v = np.zeros((n_lev,n_lat,n_lon))
   t = np.zeros((n_lev,n_lat,n_lon))
   q = np.zeros((n_lev,n_lat,n_lon))
   z = np.zeros((n_lev,n_lat,n_lon))

   #Read in the arrays; aida_data goes from 90 to -90
   for i in range(n_lev):
      t[i,:,:-1] = aida_out[0,::-1,:,n_lev*0+i]
      t[i,:,-1] = t[i,:,0]

      q[i,:,:-1] = aida_out[0,::-1,:,n_lev*1+i]
      q[i,:,-1] = q[i,:,0]

      z[i,:,:-1] = aida_out[0,::-1,:,n_lev*2+i]
      z[i,:,-1] = z[i,:,0]

      u[i,:,:-1] = aida_out[0,::-1,:,n_lev*3+i]
      u[i,:,-1] = u[i,:,0]

      v[i,:,:-1] = aida_out[0,::-1,:,n_lev*4+i]
      v[i,:,-1] = v[i,:,0]

   aida_fh.close()

   #Create a dict with all of the AIDA data in it
   aida_data = { 'n_lev' : n_lev,
            'n_lon' : n_lon,
            'n_lat' : n_lat,
            'lat' : lat,
            'lon' : lon,
            'p' : p,
            'u' : u,
            'v' : v,
            't' : t,
            'q' : q,
            'z' : z }

   return aida_data

def aida2gfs(aida_data, gfs_fname):
   #Open the GFS netCDF file
   gfs_fh = nc.Dataset(gfs_fname, 'r+')

   #Get the three GFS grids (center, S, W)
   gfs_lat = np.array(gfs_fh.variables["geolat"])
   gfs_lon = np.array(gfs_fh.variables["geolon"])
   gfs_lat_w = np.array(gfs_fh.variables["geolat_w"])
   gfs_lon_w = np.array(gfs_fh.variables["geolon_w"])
   gfs_lat_s = np.array(gfs_fh.variables["geolat_s"])
   gfs_lon_s = np.array(gfs_fh.variables["geolon_s"])

   ####
   #Regrid the AIDA data to the appropriate GFS grids
   ####

   #Create regridded arrays (_r for regridded)
   n_lev = aida_data['n_lev']
   aida_lat = aida_data['lat']
   aida_lon = aida_data['lon']
   t_r = []
   q_r = []
   z_r = []
   uw_r = []
   vw_r = []
   us_r = []
   vs_r = []

   #Perform the regridding, one AIDA-level at a time, for each variable
   for i in range(n_lev):
      #TODO When running on a node, this could be split into 7 parallel tasks
      t_r.append(interp_gfs_2_aida(gfs_lat,gfs_lon,aida_lat,aida_lon,aida_data['t'][i,:,:]))
      q_r.append(interp_gfs_2_aida(gfs_lat,gfs_lon,aida_lat,aida_lon,aida_data['q'][i,:,:]))
      z_r.append(interp_gfs_2_aida(gfs_lat,gfs_lon,aida_lat,aida_lon,aida_data['z'][i,:,:]))
      uw_r.append(interp_gfs_2_aida(gfs_lat_w,gfs_lon,aida_lat,aida_lon,aida_data['u'][i,:,:]))
      vw_r.append(interp_gfs_2_aida(gfs_lat_w,gfs_lon,aida_lat,aida_lon,aida_data['v'][i,:,:]))
      us_r.append(interp_gfs_2_aida(gfs_lat_s,gfs_lon,aida_lat,aida_lon,aida_data['u'][i,:,:]))
      vs_r.append(interp_gfs_2_aida(gfs_lat_s,gfs_lon,aida_lat,aida_lon,aida_data['v'][i,:,:]))

   t_r = np.array(t_r)
   q_r = np.array(q_r)
   z_r = np.array(z_r)
   uw_r = np.array(uw_r)
   vw_r = np.array(vw_r)
   us_r = np.array(us_r)
   vs_r = np.array(vs_r)

   ####
   #Perform vertical interpolation
   ####

   #Construct pressure from surface pressure and del(pressure)
   gfs_ps = np.array(gfs_fh.variables["ps"])
   gfs_delp = np.array(gfs_fh.variables["delp"])
   gfs_p = np.zeros(gfs_delp.shape)
   gfs_p[0,:,:] = gfs_ps[:,:] - gfs_delp[0,:,:]

   for lev in range(1,gfs_delp.shape[0]-1):
      gfs_p[lev,:,:] = gfs_p[lev-1,:,:] - gfs_delp[lev,:,:]

   #Model top is 0mb -- using gfs_delp to calculate this will result in
   #something else due to roundoff error
   gfs_p[-1,:,:] = 0.0

   #Interpolate up to AIDA top on gfs vertical grid
   t_i = vert_interp(t_r, gfs_p, np.array(aida_data['p']))
   q_i = vert_interp(q_r, gfs_p, np.array(aida_data['p']))
   z_i = vert_interp(z_r, gfs_p, np.array(aida_data['p']))
   uw_i = vert_interp(uw_r, gfs_p, np.array(aida_data['p']))
   vw_i = vert_interp(vw_r, gfs_p, np.array(aida_data['p']))
   us_i = vert_interp(us_r, gfs_p, np.array(aida_data['p']))
   vs_i = vert_interp(vs_r, gfs_p, np.array(aida_data['p']))

   out = nc.Dataset("out_gfs.nc", 'w')
   lev = out.createDimension("lev", 128)
   aida_lev = out.createDimension("aida_lev", n_lev)
   levp = out.createDimension("levp", 129)
   lat = out.createDimension("lat", 384)
   lon = out.createDimension("lon", 384)
   latp = out.createDimension("latp", 385)
   lonp = out.createDimension("lonp", 385)

   geolat = out.createVariable("geolat","f4",("lat","lon"))
   geolatw = out.createVariable("geolatw","f4",("lat","lonp"))
   geolats = out.createVariable("geolats","f4",("latp","lon"))
   geolon = out.createVariable("geolon","f4",("lat","lon"))
   geolonw = out.createVariable("geolonw","f4",("lat","lonp"))
   geolons = out.createVariable("geolons","f4",("latp","lon"))
   geolat[:,:] = gfs_lat
   geolatw[:,:] = gfs_lat_w
   geolats[:,:] = gfs_lat_s
   geolon[:,:] = gfs_lon
   geolonw[:,:] = gfs_lon_w
   geolons[:,:] = gfs_lon_s

   p = out.createVariable("ps","f8",("lev","lat","lon"))
   t = out.createVariable("t","f8",("lev","lat","lon"))
   t_aida = out.createVariable("t_aida","f8",("aida_lev","lat","lon"))
   p[:,:,:] = gfs_p
   t[:,:,:] = t_i
   t_aida[:,:,:] = t_r
   #delp = out.createVariable("delp","f8",("lev","lat","lon"))
   #zh = out.createVariable("zh","f8",("levp","lat","lon"))
   #u_w = out.createVariable("u_w","f8",("lev","lat","lonp"))
   #v_w = out.createVariable("v_w","f8",("lev","lat","lonp"))
   #u_s = out.createVariable("u_s","f8",("lev","latp","lon"))
   #v_s = out.createVariable("v_s","f8",("lev","latp","lon"))

   out.close()

def interp_gfs_2_aida(gfs_lat,gfs_lon,lat,lon,X):

   #Transpose GFS lat/lon
   Xt = np.transpose(X)
   gfs_lont = np.transpose(gfs_lon)
   gfs_latt = np.transpose(gfs_lat)

   #Create an interpolation function
   print(lon.shape, lat.shape, Xt.shape)
   f = interp2d(lon, lat, X, kind='linear')

   #There has to be a more efficient way to do this.  I just haven't figured it out yet
   #Interpolate the AIDA data to the GFS grid points
   print(gfs_lon.min(), gfs_lon.max(), gfs_lat.min(), gfs_lat.max())
   print(lon.min(),lon.max(),lat.min(),lat.max())
   print(X.min(), X.max())
   X_gfs = np.zeros(gfs_lon.shape)
   for i in range(gfs_lon.shape[0]):
      for j in range(gfs_lon.shape[1]):
         X_gfs[i, j] = f(gfs_lon[i,j], gfs_lat[i,j])

   return X_gfs

def vert_interp(X_in,p,aida_p):

   #Calculate log(p) for AIDA and gfs
   #ptop is 0mb, so just set it to a large negative
   gfs_logp = np.zeros(p.shape)
   gfs_logp[0:-1,:,:] = np.log(p[0:-1,:,:])
   gfs_logp[-1,:,:] = -999.0

   aida_logp = np.log(aida_p)

   #Assign AIDA indices to each GFS point
   n_aida = len(aida_p)
   AIDA_ndx = np.zeros(p.shape) - 1
   for i in range(n_aida-1):
      AIDA_ndx = np.where(AIDA_ndx == -1, np.where((p > aida_p[i+1]) & (p <= aida_p[i]), i, AIDA_ndx), AIDA_ndx)

   #Preassign interpolation pressures and Xs
   P0 = np.zeros(p.shape) - 999.0
   P1 = np.zeros(p.shape) - 999.0
   X0 = np.zeros(p.shape) - 999.0
   X1 = np.zeros(p.shape) - 999.0
   for i in range(n_aida-1):
      P0 = np.where(AIDA_ndx == i, aida_logp[i], P0)
      P1 = np.where(AIDA_ndx == i, aida_logp[i+1], P1)
      for j in range(p.shape[0]):
         a = AIDA_ndx[j,:,:]
         a = X_in[i,:,:]
         a = X0[j,:,:]
         X0[j,:,:] = np.where(AIDA_ndx[j,:,:] == i, X_in[i,:,:], X0[j,:,:])
         X1[j,:,:] = np.where(AIDA_ndx[j,:,:] == i, X_in[i+1,:,:], X1[j,:,:])

   #Interpolate the quantity for each GFS point, stopping at AIDA top
   X_out = np.zeros(p.shape) - 999.0
   valid = np.where(AIDA_ndx != -1)
   X_out[valid] = X0[valid] + ((X1[valid] - X0[valid]) * (gfs_logp[valid] - P0[valid])) / (P1[valid] - P0[valid])

   return X_out


