#!/bin/bash

# Define the paths
miranda_path=~/Desktop/Data/raw/miranda/

# Define the light parameters
miranda_light="-sh -0.59 -0.89 0.65 1.0 0.17"

# Define the camera parameters
miranda_camera="-c 623.944702 186.757248 -223.472321 623.200317 186.759384 -222.804565 -0.667761 -0.002374 -0.744372 45.000000"

# Define the mcSizes array
miranda_mcSizes=("384 384 256" "192 192 128" "96 96 64" "48 48 32" "24 24 16")

# Define the file pairs arrays
miranda_file_pairs=(
    "density_384x384x256_double64.raw tfn_state_0.tf"
    "velmag_384x384x256_double64.raw tfn_state_1.tf"
    "diffusivity_384x384x256_double64.raw tfn_state_2.tf"
    "pressure_384x384x256_double64.raw tfn_state_3.tf"
)

# Redirect output to mcSizeExp.txt
exec > mirandaMcSizeExp.txt

# Iterate over values of -m from 0 to 2
for render_mode in {0..3}; do
    # Iterate over mcSizes array
    for mc_size in "${miranda_mcSizes[@]}"; do
        # Iterate over file pairs
        for index in {0..4}; do
            
            # Set the output filename to the index
            output_filename="miranda_$index"
            
            # Check if it's the 5th execution and run with all file pairs
            if [ "$index" -eq 4 ]; then
                # Run with light parameters, mcSizes, all file pairs, and output filename
                ../build/dTViewer -fr "${miranda_path}density_384x384x256_double64.raw" \
                           "${miranda_path}velmag_384x384x256_double64.raw" \
                           "${miranda_path}diffusivity_384x384x256_double64.raw" \
                           "${miranda_path}pressure_384x384x256_double64.raw" \
                           -t "${miranda_path}tfn_state_0.tf" \
                           "${miranda_path}tfn_state_1.tf" \
                           "${miranda_path}tfn_state_2.tf" \
                           "${miranda_path}tfn_state_3.tf" \
                           -r 1024 1024 $miranda_light -bg 0.3 0.3 0.3 -cb -dt 0.17 $miranda_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"

                # Run without light parameters, mcSizes, all file pairs, and output filename
                ../build/dTViewer -fr "${miranda_path}density_384x384x256_double64.raw" \
                           "${miranda_path}velmag_384x384x256_double64.raw" \
                           "${miranda_path}diffusivity_384x384x256_double64.raw" \
                           "${miranda_path}pressure_384x384x256_double64.raw" \
                           -t "${miranda_path}tfn_state_0.tf" \
                           "${miranda_path}tfn_state_1.tf" \
                           "${miranda_path}tfn_state_2.tf" \
                           "${miranda_path}tfn_state_3.tf" \
                           -r 1024 1024 -bg 0.3 0.3 0.3 -cb -dt 0.17 $miranda_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"
            else
	    	file_pair="${miranda_file_pairs[$index]}"
	    
	    	# Split file pair into two variables
	    	read -r fr_param t_param <<< "$file_pair"
                # Run with light parameters, mcSizes, file pair, and output filename
                ../build/dTViewer -fr "${miranda_path}$fr_param" \
                           -t "${miranda_path}$t_param" \
                           -r 1024 1024 $miranda_light -bg 0.3 0.3 0.3 -cb -dt 0.17 $miranda_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"

                # Run without light parameters, mcSizes, file pair, and output filename
                ../build/dTViewer -fr "${miranda_path}$fr_param" \
                           -t "${miranda_path}$t_param" \
                           -r 1024 1024 -bg 0.3 0.3 0.3 -cb -dt 0.17 $miranda_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"
            fi
        done
    done
done

