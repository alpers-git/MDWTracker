#!/bin/bash

# Define the paths
nyx_path=~/Desktop/Data/raw/nyx/

# Define the camera parameters
nyx_camera="-c -0.202679 0.919677 0.926256 0.346511 0.638234 0.139371 0.176116 0.959416 -0.220236 45.000000"

# Define the file pairs arrays
nyx_file_pairs=(
    "baryon_512x512x512_float32.raw tfn_state_0.tf"
    "dark_matter_512x512x512_float32.raw tfn_state_1.tf"
    "temperature_512x512x512_float32.raw tfn_state_2.tf"
    "velmag_512x512x512_float32.raw tfn_state_3.tf"
)

# Redirect output (stdout and stderr) to nyxRenderModes.txt
exec > nyxBlendExps.txt

for ((i = 0; i < ${#nyx_file_pairs[@]}; i++)); do
	file_pair=(${nyx_file_pairs[$i]})
        fr_params+="${nyx_path}${file_pair[0]} "
        t_params+="${nyx_path}/blendingTF/${file_pair[1]} "
done
# Loop over specified render modes
for m_mode in 2 4 5 6 8 9; do

    # Run the command
    ../build/dTViewer -fr $fr_params -t $t_params \
        -bg 0.0 0.0 0.0 -r 1024 1024 $nyx_camera -m "$m_mode" -dt 0.000280 -n 500 -mc 128 128 28 -o nyx
done

