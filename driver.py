from aida2gfs import get_aida, aida2gfs
import numpy as np
from os import path

aida_data = get_aida("in/exp004murz_pernak_predict_PREP_test_for_GFS.nc")

in_path = "in"
ctrl_fname = "gfs_ctrl.nc"

for i in range(1,7):
   gfs_fname = path.join(in_path,"gfs_data.tile"+str(i)+".nc")
   aida2gfs(aida_data, fname, ctrl_fname, debug="yes")
