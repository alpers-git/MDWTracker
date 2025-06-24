#!/bin/bash

# Define the paths
zebrafish_path=~/Desktop/Data/raw/zebrafish/

# Define the light parameters
zebrafish_light="-sh -0.34 1.0 -1.0 1.260 0.0"

# Define the camera parameters
zebrafish_camera="-c 3.587413 2.328304 2.037139 2.826688 2.063135 1.444686 -0.559435 -0.195004 0.805609 45.000000"

# Define the file pairs arrays
zebrafish_file_pairs=(
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw teaser_0.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw teaser_1.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw teaser_2.tf"
)

# Redirect output to zebrafishCustomExp.txt
exec > zebrafishCustomExp.txt

# Execute with -m 1 and without -mc parameters for each channel
for index in {0..2}; do
    file_pair="${zebrafish_file_pairs[$index]}"
    
    # Split file pair into two variables
    read -r fr_param t_param <<< "$file_pair"
    
    # Set the output filename for individual channels
    output_filename="zebrafish_channel_$index"
    
    # Run with light parameters, file pair, output filename, and -m 1
    ./build/mdwtViewer -fr "${zebrafish_path}$fr_param" \
               -t "${zebrafish_path}$t_param" \
               -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
               -o "$output_filename" -m 2 -wu 25 -n 500 -nt 500 -mc 80 80 15
    # Run with light parameters, file pair, output filename, and -m 1
    ./build/mdwtViewer -fr "${zebrafish_path}$fr_param" \
               -t "${zebrafish_path}$t_param" \
               -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
               -o "$output_filename" -m 2 -wu 25 -n 1000 -hm 1e16 -mc 80 80 15
done

# Run the last execution with all channels together
output_filename="zebrafish_all_channels"
./build/mdwtViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
           -t "${zebrafish_path}teaser_0.tf" \
           "${zebrafish_path}teaser_1.tf" \
           "${zebrafish_path}teaser_2.tf" \
           -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
           -o "$output_filename" -m 2 -wu 25 -n 500 -nt 500 -mc 80 80 15
           
./build/mdwtViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
           -t "${zebrafish_path}teaser_0.tf" \
           "${zebrafish_path}teaser_1.tf" \
           "${zebrafish_path}teaser_2.tf" \
           -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
           -o "$output_filename" -m 2 -wu 25 -n 1000 -hm 1e16 -mc 80 80 15

./build/mdwtViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
           -t "${zebrafish_path}teaser_0.tf" \
           "${zebrafish_path}teaser_1.tf" \
           "${zebrafish_path}teaser_2.tf" \
           -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
           -o "$output_filename" -m 4 -wu 25 -n 500 -mc 80 80 15
           
./build/mdwtViewer -fr "${zebrafish_path}isl1actinTop3_D_channel_1_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_2_640x640x121_uint8.raw" \
           "${zebrafish_path}isl1actinTop3_D_channel_3_640x640x121_uint8.raw" \
           -t "${zebrafish_path}teaser_0.tf" \
           "${zebrafish_path}teaser_1.tf" \
           "${zebrafish_path}teaser_2.tf" \
           -cb 3.0 3.0 1.25 -r 2560 1440 $zebrafish_light -dt 0.012 -bg 0.3 0.3 0.3 $zebrafish_camera \
           -o "$output_filename" -m 5 -wu 25 -n 500 -mc 80 80 15
