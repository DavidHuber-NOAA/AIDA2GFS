#!/usr/bin/bash

fcst_dir="/scratch1/NESDIS/nesdis-rdo2/David.Huber/archive/aida_p0"
anl_dir="/scratch1/NCEPDEV/global/Mallory.Row/archive/gfs"
fcst_tmpl="pgbfHH.gfs.PDYCYC.grib2"
anl_tmpl="pgbanl.gfs.PDYCYC.grib2"
days2compare="1 3 5 7 10 14 15"

stmp=$1
name_fcst=$2
name_anl=$3
fcst_init=$4 #YYYYMMDDCC

fcst_init_PDY=$(echo $fcst_init | cut -b 1-8)
fcst_init_cyc=$(echo $fcst_init | cut -b 9-10)

echo mkdir $stmp
mkdir -p $stmp

set -eu
for var in "wind" "hgt" "rh"; do
   cfg="${var}_diff.cfg"
   NAME=$(echo ${var} | tr 'a-z' 'A-Z')

   for d in days2compare; do
      hr=$((d*24))
      fcst_PDYcyc=$(ndate $hr $fcst_init)
      fcst_PDY=$(echo $fcst_PDYcyc | cut -b 1-8)
      fcst_cyc=$(echo $fcst_PDYcyc | cut -b 9-10)
      l_date="${hr}0000L_${fcst_PDY}_${fcst_cyc}0000V"
      fcst_file=$(echo $fcst_tmpl | sed "s/HH/$hr/" | sed "s/PDY/$fcst_init/")
      fcst_file=${fcst_dir}/${fcst_file}

      anl_file=$(echo $anl_tmpl | sed "s/PDY/$fcst_PDY/" | sed "s/CYC/$fcst_cyc/")
      anl_file=${anl_dir}/${anl_file}

      grid_stat ${fcst_file} ${anl_file} ${cfg} -outdir ${stmp} -v 2
      nc_out="${stmp}/grid_stat_${l_date}_pairs.nc"

      for lev in "200" "500" "700"; do
         field="DIFF_${NAME}_P${lev}_${NAME}_P${lev}_FULL"
         cfg_string='name=''"'${field}'"'';level=''"'P${lev}'"'';'
         range_string="-plot_range -70 70"

         ps_name="gsi_vs_aida/d${d}/diff_${name_fcst}_${lev}mb_${var}_d${d}.ps"
         png_name="gsi_vs_aida/d${d}/diff_${name_fcst}_${lev}mb_${var}_d${d}.png"
         plot_data_plane ${nc_out} ${ps_name} ${cfg_string} $range_string
         convert -rotate 90 -trim ${ps_name} ${png_name}
      done
   done

done



