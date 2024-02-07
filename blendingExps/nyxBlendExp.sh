#!/bin/bash

# Define the paths
nyx_path=~/Desktop/Data/raw/nyx/

# Define the camera parameters
nyx_camera="-c -0.269169 0.774701 0.732508 0.486144 0.601800 0.100359 0.114627 0.984556 -0.132329 45.000000"

# Define the file pairs arrays
nyx_file_pairs=(
    "baryon_512x512x512_float32.raw tfn_state_0.tf"
    "dark_matter_512x512x512_float32.raw tfn_state_1.tf"
    "temperature_512x512x512_float32.raw tfn_state_2.tf"
    "velmag_512x512x512_float32.raw tfn_state_3.tf"
)

# Redirect output (stdout and stderr) to nyxRenderModes.txt
exec > nyxBlendExps.txt 2>&1

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

