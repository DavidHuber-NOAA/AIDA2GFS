from aida2gfs import get_aida, aida2gfs
import numpy as np
from os import path

in_path = "in"
ctrl_fname = path.join(in_path,"gfs_ctrl.nc")
aida_fname = path.join(in_path,"exp004murz_pernak_predict_PREP_test_for_GFS.nc")
aida_data = get_aida(aida_fname, convert_rh="yes")

for i in range(1,7):
   gfs_fname = path.join(in_path,"gfs_data.tile"+str(i)+".nc")
   aida2gfs(aida_data, gfs_fname, ctrl_fname, debug="yes")
