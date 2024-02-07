#!/bin/bash

# Define the paths
zebrafish_path=~/Desktop/Data/raw/zebrafish/

# Define the light parameters
zebrafish_light="-sh 0.79 -0.72 -0.94 1.0 0.14"

# Define the camera parameters
zebrafish_camera="-c -1.085637 2.037609 3.672860 -0.464879 1.926233 2.896807 0.069570 0.993778 -0.086975 45.000000"

# Define the file pairs arrays
zebrafish_file_pairs=(
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_0.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_1.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_2.tf"
)

# Define the mcSizes array
zebrafish_mcSizes=("640 640 121" "320 320 60" "160 160 30" "80 80 15" "40 40 8")

# Redirect output to zebrafishExp.txt
exec > zebrafishMcSizeExp.txt

# Iterate over values of -m from 0 to 2
for render_mode in {0..2}; do
    # Iterate over mcSizes array
    for mc_size in "${zebrafish_mcSizes[@]}"; do
        # Iterate over file pairs
        for index in {0..3}; do
            
             	file_pair="${zebrafish_file_pairs[$index]}"
        
		# Split file pair into two variables
		read -r fr_param t_param <<< "$file_pair"
		
		# Set the output filename to the index
		output_filename="zebrafish_$index"
		
		# Check if it's the 3rd execution and run with all file pairs
		if [ "$index" -eq 3 ]; then
		    # Run with light parameters, file pairs, and output filename
		    ../build/dTViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
			       "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
			       "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
			       -t "${zebrafish_path}tfn_state_0.tf" \
			       "${zebrafish_path}tfn_state_1.tf" \
			       "${zebrafish_path}tfn_state_2.tf" \
			       -cb 3.0 3.0 1.25 -r 1024 1024 $zebrafish_light -dt 0.016 -bg 0.3 0.3 0.3 $zebrafish_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename" -mc $mc_size
		     # Run without light parameters, file pairs, and output filename
		    ../build/dTViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
			       "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
			       "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
			       -t "${zebrafish_path}tfn_state_0.tf" \
			       "${zebrafish_path}tfn_state_1.tf" \
			       "${zebrafish_path}tfn_state_2.tf" \
			       -cb 3.0 3.0 1.25 -r 1024 1024 -dt 0.016 -bg 0.3 0.3 0.3 $zebrafish_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename" -mc $mc_size

		else
		    # Run with light parameters, file pair, and output filename
		    ../build/dTViewer -fr "${zebrafish_path}$fr_param" \
			       -t "${zebrafish_path}$t_param" \
			       -cb 3.0 3.0 1.25 -r 1024 1024 $zebrafish_light -dt 0.016 -bg 0.3 0.3 0.3 $zebrafish_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename" -mc $mc_size
		    # Run without light parameters, file pair, and output filename
		    ../build/dTViewer -fr "${zebrafish_path}$fr_param" \
			       -t "${zebrafish_path}$t_param" \
			       -cb 3.0 3.0 1.25 -r 1024 1024 -dt 0.016 -bg 0.3 0.3 0.3 $zebrafish_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -o "$output_filename" -mc $mc_size
		fi
        done
    done
done
