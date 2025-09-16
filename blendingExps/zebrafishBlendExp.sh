#!/bin/bash

# Define the paths
zebrafish_path=~/Desktop/Data/raw/zebrafish/

# Define the camera parameters
zebrafish_camera="-c 0.103061 1.168218 2.507351 0.624446 1.218966 1.655540 -0.011490 0.998557 0.052457 45.000000"

# Define the file pairs arrays
zebrafish_file_pairs=(
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_0.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_1.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_2.tf"
)

# Redirect output (stdout and stderr) to zebrafishRenderModes.txt
exec > zebrafishBlendExps.txt 2>&1

for ((i = 0; i < ${#zebrafish_file_pairs[@]}; i++)); do
	file_pair=(${zebrafish_file_pairs[$i]})
        fr_params+="${zebrafish_path}${file_pair[0]} "
        t_params+="${zebrafish_path}/blendingTF/${file_pair[1]} "
done
# Loop over specified render modes
for m_mode in 2 4 5 6 8 9; do

    # Run the command
    ../build/dTViewer -fr $fr_params -t $t_params \
        -cb 3.0 3.0 1.25 -bg 0.0 0.0 0.0 -r 1024 1024 $zebrafish_camera -m "$m_mode" -dt 0.00100 -n 500 -mc 128 128 28 -o zebrafish
done

