usage="This tool uses the METplus tool plot_data_plane to plot data from a GRIB or GRIB2 file.
Example: ./plot_plane.sh PGB003.grb2 plot_dir <optional prefix>\n"

file=$1
plot_dir=$2
prefix=$3

if [[ $# -lt 2 ]]; then
   printf "$usage"
   exit
fi

#hgt=("1000" "925" "850" "700" "500" "200" "100" "10")
hgt=("700" "500" "200")

var=("PRATE" "TMP" "HGT" "UGRD" "VGRD")

for j in ${!var[@]}; do
   if [[ ${var[j]} == "PRATE" ]]; then
      x=${var[$j]}
      if [ -z ${prefix+x} ]; then
         plot=$(echo "${plot_dir}/${x}")
      else
         plot=$(echo "${plot_dir}/${prefix}_${x}")
      fi
      plot_data_plane $file ${plot}.ps 'name=''"'$x'"''; level=''"'Z0'"'';' \
         -title "$prefix ${x} at ${h}mb"
      convert -rotate 90 -trim ${plot}.ps ${plot}.png
      rm ${plot}.ps
      continue
   fi
   for i in ${!hgt[@]}; do
      x=${var[$j]}
      h=${hgt[$i]}
      if [ -z ${prefix+x} ]; then
         plot=$(echo "${plot_dir}/${h}mb_${x}")
      else
         plot=$(echo "${plot_dir}/${prefix}_${h}mb_${x}")
      fi
      plot_data_plane $file ${plot}.ps 'name=''"'$x'"''; level=''"'P$h'"'';' \
         -title "$prefix ${x} at ${h}mb"
      convert -rotate 90 -trim ${plot}.ps ${plot}.png
      rm ${plot}.ps
   done
done
