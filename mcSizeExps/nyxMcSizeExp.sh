#!/bin/bash

# Define the paths
nyx_path=~/Desktop/Data/raw/nyx/

# Define the light parameters
nyx_light="-sh 0.36 -0.5 -0.15 1.0 0.15"

# Define the camera parameters
nyx_camera="-c -0.655885 1.505893 1.667711 -0.077843 0.944703 1.075314 0.411170 0.827384 -0.382591 45.000000"

# Define the file pairs arrays
nyx_file_pairs=(
    "baryon_512x512x512_float32.raw tfn_state_0.tf"
    "dark_matter_512x512x512_float32.raw tfn_state_1.tf"
    "temperature_512x512x512_float32.raw tfn_state_2.tf"
    "velmag_512x512x512_float32.raw tfn_state_3.tf"
)

# Define the mcSizes array
nyx_mcSizes=("512 512 512" "256 256 256" "128 128 128" "64 64 64" "32 32 32")

# Redirect output to nyxExp.txt
exec > nyxMcSizeExp.txt

# Iterate over values of -m from 0 to 2
for render_mode in {0..3}; do
    # Iterate over mcSizes array
    for mc_size in "${nyx_mcSizes[@]}"; do
        # Iterate over file pairs
        for index in {0..4}; do
            file_pair="${nyx_file_pairs[$index]}"
            
            # Split file pair into two variables
            read -r fr_param t_param <<< "$file_pair"
            
            # Set the output filename to nyx and mc size
            output_filename="nyx_$index"
            # Check if it's the 5th execution and run with all file pairs
            if [ "$index" -eq 4 ]; then
		    ../build/dTViewer -fr "${nyx_path}baryon_512x512x512_float32.raw" \
		           "${nyx_path}dark_matter_512x512x512_float32.raw" \
		           "${nyx_path}temperature_512x512x512_float32.raw" \
		           "${nyx_path}velmag_512x512x512_float32.raw" \
		           -t "${nyx_path}tfn_state_0.tf" \
		           "${nyx_path}tfn_state_1.tf" \
		           "${nyx_path}tfn_state_2.tf" \
		           "${nyx_path}tfn_state_3.tf" \
		           -bg 0.3 0.3 0.3 -r 1024 1024 -mc $mc_size $nyx_light -dt 0.002900 $nyx_camera \
		           -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename"
		    
		    ../build/dTViewer -fr "${nyx_path}baryon_512x512x512_float32.raw" \
		           "${nyx_path}dark_matter_512x512x512_float32.raw" \
		           "${nyx_path}temperature_512x512x512_float32.raw" \
		           "${nyx_path}velmag_512x512x512_float32.raw" \
		           -t "${nyx_path}tfn_state_0.tf" \
		           "${nyx_path}tfn_state_1.tf" \
		           "${nyx_path}tfn_state_2.tf" \
		           "${nyx_path}tfn_state_3.tf" \
		           -bg 0.3 0.3 0.3 -r 1024 1024 -mc $mc_size -dt 0.002900 $nyx_camera \
		           -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename"
            
            else
		    # Run with light parameters, file pair, mc size, and output filename
		    ../build/dTViewer -fr "${nyx_path}$fr_param" \
		               -t "${nyx_path}$t_param" \
		               -bg 0.3 0.3 0.3 -r 1024 1024 -mc $mc_size $nyx_light -dt 0.002900 $nyx_camera \
		               -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename"
		    # Run without light parameters, file pair, mc size, and output filename
		    ../build/dTViewer -fr "${nyx_path}$fr_param" \
		               -t "${nyx_path}$t_param" \
		               -bg 0.3 0.3 0.3 -r 1024 1024 -mc $mc_size -dt 0.002900 $nyx_camera \
		               -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename"
            fi
        done
    done
done
