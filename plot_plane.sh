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

var=("WIND" "TMP" "HGT")

declare -A ranges
ranges["200HGT"]="10000 13000"
ranges["500HGT"]="4700 6000"
ranges["700HGT"]="2300 3300"
ranges["200TMP"]="195 235"
ranges["500TMP"]="220 280"
ranges["700TMP"]="200 300"
ranges["200WIND"]="0 120"
ranges["500WIND"]="0 100"
ranges["700WIND"]="0 80"
ranges["200UGRD"]="-40 85"
ranges["500UGRD"]="-40 80"
ranges["700UGRD"]="-40 50"
ranges["200VGRD"]="-50 60"
ranges["500VGRD"]="-50 60"
ranges["700VGRD"]="-40 40"

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
