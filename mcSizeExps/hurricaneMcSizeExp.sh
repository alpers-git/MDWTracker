#!/bin/bash

# Define the paths
hurricane_path=~/Desktop/Data/raw/hurricane/

# Define the light parameters
hurricane_light="-sh 0.09 -1 -1 1 0.1"

# Define the camera parameters
hurricane_camera="-c 251.565598 250.310257 716.027161 251.565598 250.310257 715.027161 0.000000 1.000000 0.000000 45.000000"

# Define the file pairs arrays
hurricane_file_pairs=(
    "CLOUDf48log10_500x500x100_float32.raw tfn_state_0.tf"
    "QRAINf48log10_500x500x100_float32.raw tfn_state_1.tf"
    "QSNOWf48log10_500x500x100_float32.raw tfn_state_2.tf"
    "QVAPORf48_500x500x100_float32.raw tfn_state_3.tf"
)

# Define the mcSizes array
hurricane_mcSizes=("500 500 100" "250 250 50" "125 125 25" "63 63 12" "32 32 6")

# Redirect output to hurricaneExp.txt
exec > hurricaneMcSizeExp.txt

# Iterate over values of -m from 0 to 2
for render_mode in {0..3}; do
    # Iterate over mcSizes array
    for mc_size in "${hurricane_mcSizes[@]}"; do
        # Iterate over file pairs
        for index in {0..4}; do
            # Set the output filename to hurricane, mc size, and index
            output_filename="hurricane_$index"
            
            if [ "$index" -eq 4 ]; then
            	# Run with light parameters, mcSizes, all file pairs, and output filename
                ../build/dTViewer -fr "${hurricane_path}CLOUDf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QRAINf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QSNOWf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QVAPORf48_500x500x100_float32.raw" \
                           -t "${hurricane_path}tfn_state_0.tf" \
                           "${hurricane_path}tfn_state_1.tf" \
                           "${hurricane_path}tfn_state_2.tf" \
                           "${hurricane_path}tfn_state_3.tf" \
                           -r 1024 1024 $hurricane_light -bg 0.3 0.3 0.3 -cb -dt 0.5 $hurricane_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"

                # Run without light parameters, mcSizes, all file pairs, and output filename
                ../build/dTViewer -fr "${hurricane_path}CLOUDf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QRAINf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QSNOWf48log10_500x500x100_float32.raw" \
                           "${hurricane_path}QVAPORf48_500x500x100_float32.raw" \
                           -t "${hurricane_path}tfn_state_0.tf" \
                           "${hurricane_path}tfn_state_1.tf" \
                           "${hurricane_path}tfn_state_2.tf" \
                           "${hurricane_path}tfn_state_3.tf" \
                           -r 1024 1024 -bg 0.3 0.3 0.3 -cb -dt 0.5 $hurricane_camera \
                           -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size -o "$output_filename"
	    else
	   	file_pair="${hurricane_file_pairs[$index]}"
	   	# Split file pair into two variables
		read -r fr_param t_param <<< "$file_pair"
		# Run with light parameters, file pair, mc size, render mode, and output filename
		../build/dTViewer -fr "${hurricane_path}$fr_param" \
			       -t "${hurricane_path}$t_param" \
			       -cb -bg 0.3 0.3 0.3 -r 1024 1024 $hurricane_light -dt 0.5 $hurricane_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size \
			       -o "$output_filename"
		 # Run without light parameters, file pair, mc size, render mode, and output filename
		 ../build/dTViewer -fr "${hurricane_path}$fr_param" \
			       -t "${hurricane_path}$t_param" \
			       -cb -bg 0.3 0.3 0.3 -r 1024 1024 -dt 0.5 $hurricane_camera \
			       -wu 25 -n 500 -nt 500 -m "$render_mode" -mc $mc_size \
			       -o "$output_filename"
	   fi
        done
    done
done
