import netCDF4 as nc
from scipy.interpolate import interp2d
import xarray as xr
import numpy as np
import xesmf as xe
from os import mkdir
from os.path import isdir

def get_aida(aida_out_fname):

   aida_fh = nc.Dataset(aida_out_fname, 'r')
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
   sphum = np.zeros((lev,lat,lon))
   zh = np.zeros((lev,lat,lon))

   #Read in the arrays; aida_data goes from 90 to -90
   for i in range(lev):
      t[i,:,:-1] = aida_out[0,::-1,:,lev*0+i]
      t[i,:,-1] = t[i,:,0]

      sphum[i,:,:-1] = aida_out[0,::-1,:,lev*1+i]
      sphum[i,:,-1] = sphum[i,:,0]

      zh[i,:,:-1] = aida_out[0,::-1,:,lev*2+i]
      zh[i,:,-1] = zh[i,:,0]

      u[i,:,:-1] = aida_out[0,::-1,:,lev*3+i]
      u[i,:,-1] = u[i,:,0]

      v[i,:,:-1] = aida_out[0,::-1,:,lev*4+i]
      v[i,:,-1] = v[i,:,0]

   aida_fh.close()

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

def get_gfs(gfs_in_fname, ctrl_fname):

   #Open the GFS input file
   gfs_fh = nc.Dataset(gfs_in_fname, 'r')

   #Open the GFS control file
   gfs_ctrl_fh = nc.Dataset(ctrl_fname, "r")
   P0 = gfs_ctrl_fh["vcoord"][0,1]

   #Get dimensions from the file and place in the function output dict
   gfs_data = {
         'lev'  : gfs_fh.dimensions["lev"].size,
         'p0'   : P0,
         'lat'  : gfs_fh.dimensions["lat"].size,
         'lon'  : gfs_fh.dimensions["lon"].size,
         'levp' : gfs_fh.dimensions["levp"].size,
         'latp' : gfs_fh.dimensions["latp"].size,
         'lonp' : gfs_fh.dimensions["lonp"].size,
         'ntracer' : gfs_fh.dimensions["ntracer"].size}

   #Read all data into numpy arrays and add to the dictionary
   for name,var in gfs_fh.variables.items():
      gfs_data[name] = np.array(var)

   #Read 'source' global attribute
   gfs_data["source"] = gfs_fh.source
   # Construct pressure and add it to the dictionary
   # delp is the thickness of each layer.  The center is pressure weighted mean at the edges:
   # p_center = (P2 - P1)/log(P2/P1), P2 > P1
   delp = gfs_data["delp"]
   # Find p_int (pressure at the top/bot boundaries) and p (p_center)
   p_int = np.zeros([delp.shape[0]+1,delp.shape[1],delp.shape[2]])
   p_int[0,:,:] = gfs_data["p0"]
   p = np.zeros(delp.shape)
   for k in range(1,p_int.shape[0]):
      p_int[k,:,:] = p_int[k-1,:,:] + delp[k-1,:,:]
      p[k-1,:,:] = (p_int[k-1,:,:] + p_int[k,:,:]) * 0.5

   #Add GFS pressures to the dict
   gfs_data["p"] = p
   gfs_data["p_int"] = p_int

   return gfs_data

def aida2gfs(aida_data, gfs_fname, ctrl_fname, debug="no", do_blend="no"):
   #Open and read the GFS file and control file
   print("Read in GFS data from " + gfs_fname)
   gfs_data = get_gfs(gfs_fname, ctrl_fname)

   ####
   #Regrid the AIDA data to the appropriate GFS grids
   ####

   print("Regrid data")
   #Create regridded arrays (_r for regridded)
   aida_geolat = aida_data['geolat']
   aida_geolon = aida_data['geolon']
   t_r = []

   #Perform the regridding, one AIDA-level at a time, for each variable
   cen_var = ["sphum", "t", "zh"]
   s_var = ["u","v"]
   w_var = ["u","v"]
   regrid_vars = {}
   #Regrid all AIDA variables to the GFS grid, one grid location at a time (center, southern, western)
   for var_set, loc in zip([cen_var, s_var, w_var], ["","_s","_w"]):
      if("_s" in loc):
         (names, variables) = regrid(gfs_data["geolat_s"],gfs_data["geolon_s"],
            aida_geolat,aida_geolon,aida_data,var_set,loc)
      elif("_w" in loc):
         (names, variables) = regrid(gfs_data["geolat_w"],gfs_data["geolon_w"],
            aida_geolat,aida_geolon,aida_data,var_set,loc)
      else:
         (names, variables) = regrid(gfs_data["geolat"],gfs_data["geolon"],
            aida_geolat,aida_geolon,aida_data,var_set,loc)

      #Populate the regridded variables dictionary
      for name, variable in zip(names,variables):
         regrid_vars[name] = np.array(variable)

   #Now regrid GFS pressure to the southern and western grids, one altitude at a time
   (dmy, [gfs_data["p_s"]]) = regrid(gfs_data["geolat_s"],gfs_data["geolon_s"],
       gfs_data["geolat"], gfs_data["geolon"], gfs_data, ["p"],"_s")

   (dmy, [gfs_data["p_w"]]) = regrid(gfs_data["geolat_w"], gfs_data["geolon_w"],
       gfs_data["geolat"], gfs_data["geolon"], gfs_data, ["p"], "_w")

   ####
   #Perform vertical interpolation
   ####

   print("Interpolate AIDA data to GFS grid")
   #Interpolate up to AIDA top on gfs vertical grid
   interp_vars = vert_interp(regrid_vars, gfs_data, aida_data['p'])

   #Blend GFS and interpolated AI-DA solutions
   if(do_blend.lower() == "yes"):
      print("Blend AIDA data at upper and lower boundaries")
      blended_vars = blend(interp_vars, gfs_data)
   else:
      #Do not blend, but do combine.  blend_range=0mb
      print("Combine AIDA and input GFS data")
      interp_vars = blend(interp_vars, gfs_data, blend_range = 0)

   if(do_blend.lower() == "yes"):
      final_vars = blended_vars
   else:
      final_vars = interp_vars

   #Copy GFS data for missing AIDA inputs
   copy_list = ["w", "sphum", "o3mr", "ice_wat", "rainwat", "snowwat", "graupel"]
   for var in copy_list:
      final_vars[var] = gfs_data[var]

   #Write out the new GFS input file
   print("Write AIDA data to a new GFS netCDF4 file")
   write_gfs(gfs_fname, gfs_data, final_vars)

   #Optionally, write out a debug file
   if(debug.lower() == "yes" or debug.lower() == "true"):
      print("Write debug file")
      if(do_blend.lower() == 'yes'):
         write_debug(gfs_fname, gfs_data, aida_data, regrid_vars, interp_vars, blended_vars, final_vars)
      else:
         write_debug(gfs_fname, gfs_data, aida_data, regrid_vars, interp_vars, final_vars, final_vars)

   return

def regrid(regrid_geolat,regrid_geolon,in_geolat,in_geolon,input_data,var_list,ext):

   #Regrids the variables in the input_data dict specified by var_list.  The
   #variables should be registered on the in_geolat and in_geolon grid and will be
   #regridded to the regrid_geolat and regrid_geolon grid by bilinear interpolation.

   out_vars = []
   out_names = [var_name + ext for var_name in var_list]

   #Save inputs

   (save_in_geolat, save_in_geolon) = (in_geolat, in_geolon)
   (save_regrid_geolat, save_regrid_geolon) = (regrid_geolat, regrid_geolon)

   #####
   #Prepare input and output grids
   #####
   #Check dimensionality of input and output grids
   if(len(regrid_geolat.shape) == len(regrid_geolon.shape) and
      len(in_geolat.shape) == len(in_geolon.shape) and
      len(regrid_geolat.shape) <= 2 and len(in_geolat.shape) <= 2):

      #Determine which method to use
      if(len(in_geolat.shape) == 1 and len(regrid_geolat.shape) == 1):
         method = "interp2d_rect2rect"
      elif(len(in_geolat.shape) == 1 and len(regrid_geolat.shape) == 2):
         #method = "interp2d_rect2curv"
         method = "xesmf_rect2curv"
      elif(len(in_geolat.shape) == 2 and len(regrid_geolat.shape) == 2):
         method = "xesmf_curv2curv"
      else:
         method = "xesmf_curv2rect"

   else:
      raise ValueError("Invalid input grid dimensionality")

   #####
   #Based on regridding method, regrid each variable, one level at a time
   #####

   if(method == "xesmf_rect2curv"):
      #If we are working with rectilinear to curvilinear, we need to convert the
      #1-d rectilinear latitude/longitude to 2-d
      curv_in_geolat = np.zeros([in_geolat.shape[0],in_geolon.shape[0]])
      curv_in_geolon = np.zeros([in_geolat.shape[0],in_geolon.shape[0]])
      for i in range(in_geolat.shape[0]):
         for j in range(in_geolon.shape[0]):
            curv_in_geolat[i,j] = in_geolat[i]
            curv_in_geolon[i,j] = in_geolon[j]
      in_geolon = curv_in_geolon
      in_geolat = curv_in_geolat
   elif(method == "xesmf_curv2rect"):
      curv_regrid_geolon = np.zeros([regrid_geolon.shape[0],regrid_geolat.shape[0]])
      curv_regrid_geolat = np.zeros([regrid_geolon.shape[0],regrid_geolat.shape[0]])
      for i in range(regrid_geolon.shape[0]):
         for j in range(regrid_geolat.shape[0]):
            curv_regrid_geolon[i,j] = regrid_geolon[i]
            curv_regrid_geolat[i,j] = regrid_geolat[j]
      regrid_geolon = curv_regrid_geolon
      regrid_geolat = curv_regrid_geolat

   if(method == "interp2d_rect2rect"):
      raise ValueError("The interp2d_rect2rect method has not been created yet")
   elif(method == "interp2d_rect2curv"):
      for var in var_list:
         X = np.zeros([input_data[var].shape[0],regrid_geolon.shape[0],regrid_geolon.shape[1]]) - 999.0
         for k in range(input_data["lev"]):
            f = interp2d(in_geolon, in_geolat, input_data[var][k,:,:], kind='linear')

            #There has to be a more efficient way to do this.  I just haven't figured it out yet.
            #Interpolate the AIDA data to the GFS grid points
            for i in range(regrid_geolon.shape[0]):
               for j in range(regrid_geolon.shape[1]):
                  X[k, i, j] = f(regrid_geolon[i,j], regrid_geolat[i,j])

      out_vars.append(X)

   elif(method == "xesmf_curv2curv" or method == "xesmf_rect2curv"):
      #Create an output grid
      dmy = np.zeros([input_data["lev"], regrid_geolat.shape[0], regrid_geolat.shape[1]])
      z = [k for k in range(input_data["lev"])]
      out_dr = xr.DataArray(dmy, coords = {'z':('z',z), "lat": (('x','y'), regrid_geolat),
                                           "lon": (('x','y'),regrid_geolon)}, dims = ('z', 'x', 'y'))
      for var in var_list:
         X = np.zeros([input_data[var].shape[0],regrid_geolon.shape[0],regrid_geolon.shape[1]]) - 999.0

         in_dr = xr.DataArray(input_data[var][:,:,:], dims = ('z','x', 'y'),
               coords = {'z':('z', z), 'lat':(('x','y'), in_geolat), 'lon':(('x','y'), in_geolon)})
         f = xe.Regridder(in_dr, out_dr, 'bilinear', extrap_method="inverse_dist")
         X[:,:,:] = f(in_dr).data

         out_vars.append(X)

   elif(method == "xesmf_curv2rect"):
      raise ValueError("The xesmf_curv2rect method has not been created yet")


   #Restore the input latitude/longitude grids

   (in_geolat, in_geolon) = (save_in_geolat, save_in_geolon)
   (regrid_geolat, regrid_geolon) = (save_regrid_geolat, save_regrid_geolon)

   return (out_names, out_vars)

def vert_interp(dict_in,gfs_data,aida_p):

   #Initialize the output dict
   dict_out = {}

   #Calculate log(p) for AIDA and gfs (at center, southern, western, and interface grid points)
   (p_c, p_s, p_w, p_i) = (np.array(gfs_data["p"]), np.array(gfs_data["p_s"]),
                           np.array(gfs_data["p_w"]), np.array(gfs_data["p_int"]))
   #Now calculate logs for each pressure set
   (gfs_logp_c, gfs_logp_s, gfs_logp_w, gfs_logp_i) = (np.zeros(p_c.shape), np.zeros(p_s.shape), np.zeros(p_w.shape), np.zeros(p_i.shape))
   gfs_logp_s[1:,:,:] = np.log(p_s[1:,:,:])
   gfs_logp_w[1:,:,:] = np.log(p_w[1:,:,:])
   gfs_logp_c[1:,:,:] = np.log(p_c[1:,:,:])
   gfs_logp_i[1:,:,:] = np.log(p_i[1:,:,:])
   #ptop is 0mb, so just set it to a large negative
   gfs_logp_c[0,:,:] = -999.0
   gfs_logp_s[0,:,:] = -999.0
   gfs_logp_w[0,:,:] = -999.0
   gfs_logp_i[0,:,:] = -999.0

   #Now take the log of AI-DA's pressure
   aida_logp = np.log(aida_p)

   #Assign AIDA indices to each GFS point; i.e. what AIDA point is below each GFS point based on pressure
   n_aida = len(aida_p)

   #Iterate through the input 3d arrays and interpolate to the GFS vertical grid
   for key, X_in in dict_in.items():
      if("_s" in key):
         p = p_s
         gfs_logp = gfs_logp_s
      elif("_w" in key):
         p = p_w
         gfs_logp = gfs_logp_w
      elif(key == "zh"):
         p = p_i
         gfs_logp = gfs_logp_i
      else:
         p = p_c
         gfs_logp = gfs_logp_c

      AIDA_ndx = np.zeros(p.shape) - 1
      for i in range(n_aida-2,-1,-1):
         AIDA_ndx = np.where(AIDA_ndx == -1, np.where((p > aida_p[i]) & (p <= aida_p[i+1]), i, AIDA_ndx), AIDA_ndx)

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


def blend(interp_vars,gfs_data,blend_range=5000):

   #The column data in X_aida will be like [-999,...,-999,valid,...,valid,-999,...]
   #We will blend the input GFS and AIDA solutions from the first valid point
   #to X Pa (5000 by default) above that point.  Similarly, blend the GFS and AIDA solutions from the
   #top valid point to blend_range Pa below.  Lastly, assign all invalid (-999) points with
   #the GFS solution.
   blended_vars = {}
   for key, X_aida in interp_vars.items():
      (d0, d1, d2) = X_aida.shape
      if("_w" in key):
         p = gfs_data['p_w']
      elif("_s" in key):
         p = gfs_data['p_s']
      else:
         p = gfs_data['p']
      X_out = X_aida
      #Do not blend geopotential height
      #TODO Estimate pressure at zh (half) levels
      if(key == "zh"):
         blended_vars['zh'] = X_out
         continue
      X_gfs = gfs_data[key]
      if blend_range > 0:
         for i in range(d1):
            for j in range(d2):
               #Find the top and bottom valid indices for each grid point
               bot = np.where(X_aida[:,i,j] != -999.0)[0][-1]
               top = np.where(X_aida[:,i,j] != -999.0)[0][0]

               #Blend the bottom to blend_range Pa above
               p_sub_bot = p[:bot,i,j] - p[bot,i,j] + blend_range
               bot_top = np.where(p_sub_bot > 0.0 )[0][0]

               for k in range(bot_top,bot):
                  X_out[k,i,j] = (X_aida[k,i,j] * (p[bot,i,j] - p[k,i,j]) + 
                               X_gfs[k,i,j] * (blend_range - (p[bot,i,j] - p[k,i,j]))) / blend_range

               p_sub_top = p[top+1:,i,j] - p[top,i,j] - blend_range
               top_bot = np.where(p_sub_top < 0.0)[0][-1] + top + 1
               for k in range(top,top_bot+1):
                  X_out[k,i,j] = (X_aida[k,i,j] * (p[k,i,j] - p[top,i,j]) + 
                               X_gfs[k,i,j] * (blend_range - (p[k,i,j] - p[top,i,j]))) / blend_range

      #Lastly, fill in the remaining locations with the GFS solution
      X_out = np.where(X_out == -999.0, X_gfs, X_out)

      blended_vars[key] = X_out

   return blended_vars

def write_gfs(in_fname, gfs_data, new_data):

   fname = in_fname.split("/")[-1]

   if not isdir("out"):
      mkdir("out")
   out = nc.Dataset("out/" + fname, 'w')

   #Create dimensions
   lon = out.createDimension("lon", gfs_data["lon"])
   lat = out.createDimension("lat", gfs_data["lat"])
   lonp = out.createDimension("lonp", gfs_data["lonp"])
   latp = out.createDimension("latp", gfs_data["latp"])
   lev = out.createDimension("lev", gfs_data["lev"])
   levp = out.createDimension("levp", gfs_data["levp"])
   ntracer = out.createDimension("ntracer", gfs_data["ntracer"])

   #GFS lat/lon grids
   geolon = out.createVariable("geolon","f4",("lat","lon"))
   geolat = out.createVariable("geolat","f4",("lat","lon"))
   geolon_s = out.createVariable("geolon_s","f4",("latp","lon"))
   geolat_s = out.createVariable("geolat_s","f4",("latp","lon"))
   geolon_w = out.createVariable("geolon_w","f4",("lat","lonp"))
   geolat_w = out.createVariable("geolat_w","f4",("lat","lonp"))
   geolat[:,:] = gfs_data["geolat"]
   geolat_w[:,:] = gfs_data["geolat_w"]
   geolat_s[:,:] = gfs_data["geolat_s"]
   geolon[:,:] = gfs_data["geolon"]
   geolon_w[:,:] = gfs_data["geolon_w"]
   geolon_s[:,:] = gfs_data["geolon_s"]
   [geolat.long_name, geolat_s.long_name, geolat_w.long_name] = ["Latitude" + att for att in ["", "_s", "_w"]]
   [geolon.long_name, geolon_s.long_name, geolon_w.long_name] = ["Longitude" + att for att in ["", "_s", "_w"]]
   [geolat.units, geolat_s.units, geolat_w.units] = ["degrees_north" for dmy in ["", "", ""]]
   [geolon.units, geolon_s.units, geolon_w.units] = ["degrees_east" for dmy in ["","",""]]

   #Create the output variables
   ps = out.createVariable("ps","f4",("lat","lon"))
   w = out.createVariable("w","f4",("lev","lat","lon"))
   zh = out.createVariable("zh","f4",("levp","lat","lon"))
   t = out.createVariable("t","f4",("lev","lat","lon"))
   delp = out.createVariable("delp","f4",("lev","lat","lon"))
   sphum = out.createVariable("sphum","f4",("lev","lat","lon"))
   liq_wat = out.createVariable("liq_wat","f4",("lev","lat","lon"))
   o3mr = out.createVariable("o3mr","f4",("lev","lat","lon"))
   ice_wat = out.createVariable("ice_wat","f4",("lev","lat","lon"))
   rainwat = out.createVariable("rainwat","f4",("lev","lat","lon"))
   snowwat = out.createVariable("snowwat","f4",("lev","lat","lon"))
   graupel = out.createVariable("graupel","f4",("lev","lat","lon"))
   u_w = out.createVariable("u_w","f4",("lev","lat","lonp"))
   v_w = out.createVariable("v_w","f4",("lev","lat","lonp"))
   u_s = out.createVariable("u_s","f4",("lev","latp","lon"))
   v_s = out.createVariable("v_s","f4",("lev","latp","lon"))

   #Populate variables with data
   ps[:,:] = gfs_data["ps"]
   delp[:,:,:] = gfs_data["delp"]
   w[:,:,:] = gfs_data["w"]
   #Geopotential height is tricky.  For now, just copy over the original GFS data.
   zh[:,:,:] = gfs_data["zh"]
   t[:,:,:] = new_data["t"]
   sphum[:,:,:] = new_data["sphum"]
   u_w[:,:,:] = new_data["u_w"]
   v_w[:,:,:] = new_data["v_w"]
   u_s[:,:,:] = new_data["u_s"]
   v_s[:,:,:] = new_data["v_s"]
   v_s[:,:,:] = new_data["v_s"]
   #The following don't have AIDA solutions yet, so just use input GFS data
   liq_wat[:,:,:] = gfs_data["liq_wat"]
   o3mr[:,:,:] = gfs_data["o3mr"]
   ice_wat[:,:,:] = gfs_data["ice_wat"]
   rainwat[:,:,:] = gfs_data["rainwat"]
   snowwat[:,:,:] = gfs_data["snowwat"]
   graupel[:,:,:] = gfs_data["graupel"]

   #Set coordinates attribute for each variable
   for var in [ps, delp, w, zh, t, sphum, liq_wat, o3mr, ice_wat, rainwat, snowwat, graupel]:
      var.coordinates = "geolon geolat"

   for var in [u_s, v_s]:
      var.coordinates = "geolon_s geolat_s"

   for var in [u_w, v_w]:
      var.coordinates = "geolon_w geolat_w"

   #Write the source global attribute
   out.source = gfs_data["source"]

   out.close()

   return

def write_debug(in_fname, gfs_data, aida_data, regrid_vars, interp_vars, blended_vars, final_vars):

   tile_ext = in_fname.split(".")[-2]
   out = nc.Dataset("debug/debug." + tile_ext + ".nc", 'w')
   gfs_lev = out.createDimension("lev", gfs_data["lev"])
   gfs_levp = out.createDimension("levp", gfs_data["levp"])

   lat = out.createDimension("lat", gfs_data["lat"])
   lon = out.createDimension("lon", gfs_data["lon"])
   latp = out.createDimension("latp", gfs_data["latp"])
   lonp = out.createDimension("lonp", gfs_data["lonp"])

   #GFS lat/lon grids
   geolat = out.createVariable("geolat","f4",("lat","lon"))
   geolat_w = out.createVariable("geolat_w","f4",("lat","lonp"))
   geolat_s = out.createVariable("geolat_s","f4",("latp","lon"))
   geolon = out.createVariable("geolon","f4",("lat","lon"))
   geolon_w = out.createVariable("geolon_w","f4",("lat","lonp"))
   geolon_s = out.createVariable("geolon_s","f4",("latp","lon"))
   geolat[:,:] = gfs_data["geolat"]
   geolat_w[:,:] = gfs_data["geolat_w"]
   geolat_s[:,:] = gfs_data["geolat_s"]
   geolon[:,:] = gfs_data["geolon"]
   geolon_w[:,:] = gfs_data["geolon_w"]
   geolon_s[:,:] = gfs_data["geolon_s"]

   #Original GFS surface pressure
   ps = out.createVariable("ps","f4",("lat","lon"))
   ps[:,:] = gfs_data["ps"]
   delp = out.createVariable("delp","f4",("lev","lat","lon"))
   delp[:,:,:] = gfs_data["delp"]

   #Regridded AI-DA quantities at the AI-DA pressure levels
   aida_lev = out.createDimension("aida_lev", aida_data["lev"])
   t_r = out.createVariable("t_r","f4",("aida_lev","lat","lon"))
   zh_r = out.createVariable("zh_r","f4",("aida_lev","lat","lon"))
   sphum_r = out.createVariable("sphum_r","f4",("aida_lev","lat","lon"))
   u_w_r = out.createVariable("u_w_r","f4",("aida_lev","lat","lonp"))
   v_w_r = out.createVariable("v_w_r","f4",("aida_lev","lat","lonp"))
   u_s_r = out.createVariable("u_s_r","f4",("aida_lev","latp","lon"))
   v_s_r = out.createVariable("v_s_r","f4",("aida_lev","latp","lon"))

   t_r[:,:,:] = regrid_vars["t"]
   zh_r[:,:,:] = regrid_vars["zh"]
   sphum_r[:,:,:] = regrid_vars["sphum"]
   u_w_r[:,:,:] = regrid_vars["u_w"]
   v_w_r[:,:,:] = regrid_vars["v_w"]
   u_s_r[:,:,:] = regrid_vars["u_s"]
   v_s_r[:,:,:] = regrid_vars["v_s"]

   #Regridded and vertically interpolated AI-DA quantities
   t_i = out.createVariable("t_i","f8",("lev","lat","lon"))
   zh_i = out.createVariable("zh_i","f8",("levp","lat","lon"))
   sphum_i = out.createVariable("sphum_i","f8",("lev","lat","lon"))
   u_w_i = out.createVariable("u_w_i","f8",("lev","lat","lonp"))
   v_w_i = out.createVariable("v_w_i","f8",("lev","lat","lonp"))
   u_s_i = out.createVariable("u_s_i","f8",("lev","latp","lon"))
   v_s_i = out.createVariable("v_s_i","f8",("lev","latp","lon"))

   #Geopotential height is tricky.  For now, just copy over the original GFS data.
   zh_i[:,:,:] = gfs_data["zh"]
   t_i[:,:,:] = interp_vars["t"]
   sphum_i[:,:,:] = interp_vars["sphum"]
   u_w_i[:,:,:] = interp_vars["u_w"]
   v_w_i[:,:,:] = interp_vars["v_w"]
   u_s_i[:,:,:] = interp_vars["u_s"]
   v_s_i[:,:,:] = interp_vars["v_s"]

   #Regridded, vertically interpolated, and blended AI-DA quantities
   t = out.createVariable("t","f8",("lev","lat","lon"))
   sphum = out.createVariable("sphum","f8",("lev","lat","lon"))
   zh = out.createVariable("zh","f8",("levp","lat","lon"))
   u_w = out.createVariable("u_w","f8",("lev","lat","lonp"))
   v_w = out.createVariable("v_w","f8",("lev","lat","lonp"))
   u_s = out.createVariable("u_s","f8",("lev","latp","lon"))
   v_s = out.createVariable("v_s","f8",("lev","latp","lon"))

   #Geopotential height is tricky.  For now, just copy over the original GFS data.
   zh[:,:,:] = gfs_data["zh"]
   t[:,:,:] = blended_vars["t"]
   sphum[:,:,:] = blended_vars["sphum"]
   u_w[:,:,:] = blended_vars["u_w"]
   v_w[:,:,:] = blended_vars["v_w"]
   u_s[:,:,:] = blended_vars["u_s"]
   v_s[:,:,:] = blended_vars["v_s"]

   out.close()

   return
