usage="This tool uses the METplus tool plot_data_plane to plot data from a GRIB or GRIB2 file.
Example: ./plot_plane.sh PGB003.grb2 plot_dir <optional prefix>\n"

file_fcst=$1
file_anl=$2
plot_dir=$3
prefix=$4

if [[ $# -lt 2 ]]; then
   printf "$usage"
   exit
fi

#hgt=("1000" "925" "850" "700" "500" "200" "100" "10")
hgt=("1000" "700" "500" "200")

var=("WIND" "TMP" "HGT" "RH")

declare -A ranges
ranges["200HGT"]="-500 500"
ranges["500HGT"]="-500 500"
ranges["700HGT"]="300 300"
ranges["1000HGT"]="-100 100"
ranges["200TMP"]="-20 20"
ranges["500TMP"]="-20 20"
ranges["700TMP"]="-20 20"
ranges["1000TMP"]="-20 20"
ranges["200WIND"]="-40 40"
ranges["500WIND"]="-30 30"
ranges["700WIND"]="-20 20"
ranges["1000WIND"]="-20 20"
ranges["200UGRD"]="-30 30"
ranges["500UGRD"]="-30 30"
ranges["700UGRD"]="-30 30"
ranges["1000UGRD"]="-30 30"
ranges["200VGRD"]="-20 20"
ranges["500VGRD"]="-20 20"
ranges["700VGRD"]="-20 20"
ranges["1000VGRD"]="-20 20"

for j in ${!var[@]}; do
   if [[ ${var[j]} == "PRATE" ]]; then
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
      shortname=$h$x
      if [[ "${ranges[$shortname]+abc}" ]]; then
         plot_data_plane $file ${plot}.ps 'name=''"'$x'"''; level=''"'P$h'"'';' \
            -title "$prefix ${x} at ${h}mb" -plot_range ${ranges[$shortname]}
      else
         plot_data_plane $file ${plot}.ps 'name=''"'$x'"''; level=''"'P$h'"'';' \
            -title "$prefix ${x} at ${h}mb"
      fi
      convert -rotate 90 -trim ${plot}.ps ${plot}.png
      rm ${plot}.ps
   done
done
