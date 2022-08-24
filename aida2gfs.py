import netCDF4 as nc
from scipy.interpolate import interp2d
import numpy as np

def get_aida(aida_out_fname):

   aida_fh = nc.Dataset(aida_out_fname, 'r')
   n_lev = 4

   #Read specific humidity, u, v, t, z, lat, lon
   #All are on isobaric levels (250, 350, 500, and 750mb)

   p = np.array([25000, 35000, 50000, 75000])

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

def get_gfs(gfs_in_fname):

   #Open the GFS input file
   gfs_fh = nc.Dataset(gfs_in_fname, 'r')

   #Get dimensions from the file and place in the function output dict
   gfs_data = {
         'lev'  : gfs_fh.dimensions["lev"].size,
         'lat'  : gfs_fh.dimensions["lat"].size,
         'lon'  : gfs_fh.dimensions["lon"].size,
         'levp' : gfs_fh.dimensions["levp"].size,
         'latp' : gfs_fh.dimensions["latp"].size,
         'lonp' : gfs_fh.dimensions["lonp"].size }

   #Read all data into numpy arrays and add to the dictionary
   for name,var in gfs_fh.variables.items():
      gfs_data[name] = np.array(var)

   # Construct pressure and add it to the dictionary
   delp = gfs_data["delp"]
   p = np.zeros(delp.shape)
   p[0,:,:] = gfs_data["ps"][:,:] - delp[0,:,:]

   for k in range(1,gfs_data["lev"]-1):
      p[k,:,:] = p[k-1,:,:] - delp[k,:,:]

   #Model top is 0mb -- using gfs_delp to calculate this will result in
   #something else due to roundoff error
   p[-1,:,:] = 0.0

   #Add GFS pressures to the dict
   gfs_data["p"] = p

   #Add the remaining GFS quantities; eventually these will all be replaced by AIDA
   var_names = ["zh","w", "zh", "t", "delp", "sphum", "liq_wat", "o3mr",
                "ice_wat", "rainwat", "snowwat", "graupel", "u_w", "v_w",
                "u_s", "v_s"]

   for var in var_names:
      gfs_data[var] = np.array(gfs_fh.variables[var])

   return gfs_data

def aida2gfs(aida_data, gfs_fname, debug="no"):
   #Open and read the GFS file
   gfs_data = get_gfs(gfs_fname)

   ####
   #Regrid the AIDA data to the appropriate GFS grids
   ####

   #Create regridded arrays (_r for regridded)
   n_lev = aida_data['n_lev']
   aida_lat = aida_data['lat']
   aida_lon = aida_data['lon']
   t_r = []

   #Perform the regridding, one AIDA-level at a time, for each variable
   cen_var = ["q", "z"]
   s_var = ["u","v"]
   w_var = ["u","v"]
   regrid_vars = {}
   #Regrid all variables, one grid location at a time (center, southern, western)
   for var_set, loc in zip([cen_var, s_var, w_var], ["","_s","_w"]):
      (names, variables) = regrid_aida_2_gfs(gfs_data["geolat"],gfs_data["geolon"],
            aida_lat,aida_lon,aida_data,var_set,loc)

      #Populate the regridded variables dictionary
      for name, variable in zip(names,variables):
         regrid_vars[name] = np.array(variable)

   ####
   #Perform vertical interpolation
   ####

   #Interpolate up to AIDA top on gfs vertical grid
   interp_vars = vert_interp(regrid_vars, gfs_data["p"], aida_data['p'])
   #t_i = vert_interp(t_r, gfs_p, np.array(aida_data['p']))

   #Blend GFS and interpolated AI-DA solutions
   blended_vars = blend(interp_vars, gfs_data)
   #t_b = blend(t_i, gfs_p, gfs_in_t)

   #Write out the new GFS input file
   write_gfs(gfs_fname, gfs_data)

   #Optionally, write out a debug file
   if(debug.lower() == "yes" or debug.lower() == "true"):
      write_debug(gfs_fname, gfs_data, aida_data)

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
   geolat[:,:] = gfs_data["geolat"]
   geolatw[:,:] = gfs_data["geolat_w"]
   geolats[:,:] = gfs_data["geolat_s"]
   geolon[:,:] = gfs_data["geolon"]
   geolonw[:,:] = gfs_data["geolon_w"]
   geolons[:,:] = gfs_data["geolon_s"]

   ps = out.createVariable("ps","f8",("lat","lon"))
   t = out.createVariable("t","f8",("lev","lat","lon"))
   t_aida = out.createVariable("t_aida","f8",("lev","lat","lon"))
   ps[:,:] = gfs_ps
   t[:,:,:] = t_i
   t_aida[:,:,:] = t_b
   delp = out.createVariable("delp","f8",("lev","lat","lon"))
   delp[:,:,:] = gfs_delp
   #zh = out.createVariable("zh","f8",("levp","lat","lon"))
   #u_w = out.createVariable("u_w","f8",("lev","lat","lonp"))
   #v_w = out.createVariable("v_w","f8",("lev","lat","lonp"))
   #u_s = out.createVariable("u_s","f8",("lev","latp","lon"))
   #v_s = out.createVariable("v_s","f8",("lev","latp","lon"))

   out.close()

def regrid_aida_2_gfs(gfs_lat,gfs_lon,lat,lon,aida_data,var_list,ext):

   out_vars = []
   out_names = [var_name + ext for var_name in var_list]

   #Regrid each variable, one level at a time
   for var in var_list:
      X_gfs = np.zeros([aida_data[var].shape[0],gfs_lon.shape[0],gfs_lon.shape[1]]) - 999.0
      for lev in range(aida_data["n_lev"]):
         f = interp2d(lon, lat, aida_data[var][lev,:,:], kind='linear')

         #There has to be a more efficient way to do this.  I just haven't figured it out yet.
         #Interpolate the AIDA data to the GFS grid points
         for i in range(gfs_lon.shape[0]):
            for j in range(gfs_lon.shape[1]):
               X_gfs[lev, i, j] = f(gfs_lon[i,j], gfs_lat[i,j])

      out_vars.append(X_gfs)

   return (out_names, out_vars)

def vert_interp(dict_in,p,aida_p):

   #Initialize the output dict
   dict_out = {}

   #Calculate log(p) for AIDA and gfs
   #ptop is 0mb, so just set it to a large negative
   gfs_logp = np.zeros(p.shape)
   gfs_logp[0:-1,:,:] = np.log(p[0:-1,:,:])
   gfs_logp[-1,:,:] = -999.0

   aida_logp = np.log(aida_p)

   #Assign AIDA indices to each GFS point; i.e. what AIDA point is below each GFS point based on pressure
   n_aida = len(aida_p)
   AIDA_ndx = np.zeros(p.shape) - 1
   for i in range(n_aida-2,-1,-1):
      AIDA_ndx = np.where(AIDA_ndx == -1, np.where((p > aida_p[i]) & (p <= aida_p[i+1]), i, AIDA_ndx), AIDA_ndx)

   #Iterate through the input 3d arrays and interpolate to the GFS vertical grid
   for key, X_in in dict_in.items():
      #Preassign interpolation pressures and Xs based on AIDA_ndx
      P0 = np.zeros(p.shape) - 999.0
      P1 = np.zeros(p.shape) - 999.0
      X0 = np.zeros(p.shape) - 999.0
      X1 = np.zeros(p.shape) - 999.0

      #Locate 
      for i in range(n_aida-1):
         P0 = np.where(AIDA_ndx == i, aida_logp[i], P0)
         P1 = np.where(AIDA_ndx == i, aida_logp[i+1], P1)
         for j in range(p.shape[0]):
            X0[j,:,:] = np.where(AIDA_ndx[j,:,:] == i, X_in[i,:,:], X0[j,:,:])
            X1[j,:,:] = np.where(AIDA_ndx[j,:,:] == i, X_in[i+1,:,:], X1[j,:,:])

      #Interpolate the quantity for each GFS point, stopping at AIDA top
      X_out = np.zeros(p.shape) - 999.0
      valid = np.where(AIDA_ndx != -1)
      X_out[valid] = X0[valid] + ((X1[valid] - X0[valid]) * (gfs_logp[valid] - P0[valid])) / (P1[valid] - P0[valid])

      dict_out[key] = X_out

   return dict_out


def blend(X_aida,p,X_gfs):

   #The column data in X_aida will be like [-999,...,-999,valid,...,valid,-999,...]
   #We will blend the input GFS and AIDA solutions from the first valid point
   #to 50mb above that point.  Similarly, blend the GFS and AIDA solutions from the
   #top valid point to 50mb below.  Lastly, assign all invalid (-999) points with
   #the GFS solution.
   (d0, d1, d2) = X_aida.shape
   X_out = X_aida
   for i in range(d1):
      for j in range(d2):
         #Find the top and bottom valid indices for each grid point
         bot = np.where(X_aida[:,i,j] != -999.0)[0][0]
         top = np.where(X_aida[:,i,j] != -999.0)[0][-1]

         #Blend the bottom to 50mb above
         p_sub_top = p[bot+1:,i,j] - p[bot,i,j] + 5000.0
         bot_top = np.where(p_sub_top > 0.0 )[0][-1] + bot + 1
         for k in range(bot,bot_top+1):
            X_out[k,i,j] = (X_aida[k,i,j] * (p[bot,i,j] - p[k,i,j]) + 
                            X_gfs[k,i,j] * (5000.0 - (p[bot,i,j] - p[k,i,j]))) / 5000.0

         p_sub_bot = p[:top,i,j] - p[top,i,j] - 5000.0
         top_bot = np.where(p_sub_bot < 0.0)[0][0]
         for k in range(top_bot,top+1):
            X_out[k,i,j] = (X_aida[k,i,j] * (p[k,i,j] - p[top,i,j]) + 
                            X_gfs[k,i,j] * (5000.0 - (p[k,i,j] - p[top,i,j]))) / 5000.0

   #Lastly, fill in the remaining locations with the GFS solution
   X_out = np.where(X_out == -999.0, X_gfs, X_out)

   return X_out

def write_gfs(in_fname, gfs_data):

   return

def write_debug(in_fname, gfs_data, aida_data):

   return
