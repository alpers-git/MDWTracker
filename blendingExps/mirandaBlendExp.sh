#!/bin/bash

# Define the paths
miranda_path=~/Desktop/Data/raw/miranda/

# Define the camera parameters
miranda_camera="-c 481.453247 182.770844 -20.637609 480.678436 182.764359 -20.005457 -0.632155 -0.002127 -0.774839 45.000000"

# Define the file pairs arrays
miranda_file_pairs=(
    "density_384x384x256_double64.raw tfn_state_0.tf"
    "velmag_384x384x256_double64.raw tfn_state_1.tf"
    "diffusivity_384x384x256_double64.raw tfn_state_2.tf"
    "pressure_384x384x256_double64.raw tfn_state_3.tf"
)

# Redirect output (stdout and stderr) to mirandaRenderModes.txt
exec > mirandaBlendExps.txt

for ((i = 0; i < ${#miranda_file_pairs[@]}; i++)); do
	file_pair=(${miranda_file_pairs[$i]})
        fr_params+="${miranda_path}${file_pair[0]} "
        t_params+="${miranda_path}/blendingTF/${file_pair[1]} "
done
# Loop over specified render modes
for m_mode in 2 4 5 6 8 9; do

    # Run the command
    ../build/dTViewer -fr $fr_params -t $t_params \
       -cb -bg 0.0 0.0 0.0 -r 1024 1024 $miranda_camera -m "$m_mode" -n 500 -dt 0.11 -mc 96 96 64 -o miranda
done

