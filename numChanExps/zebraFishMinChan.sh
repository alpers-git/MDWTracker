#!/bin/bash

# Define the paths
zebrafish_path=~/Desktop/Data/raw/zebrafish/

# Define the camera parameters
zebrafish_camera="-c 0.732211 3.249538 3.552723 0.956350 2.827786 2.674152 0.388831 -0.787945 0.477445 45.000000"

# Define the file pairs arrays
zebrafish_file_pairs=(
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_0.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_1.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_2.tf"
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_3.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_4.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_5.tf"
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_6.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_7.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_8.tf"
    "isl1actinTop3_D_channel_1_640x640x121_uint8.raw tfn_state_9.tf"
    "isl1actinTop3_D_channel_2_640x640x121_uint8.raw tfn_state_10.tf"
    "isl1actinTop3_D_channel_3_640x640x121_uint8.raw tfn_state_11.tf"
)

# Define the light parameters
zebrafish_light="-sh 0.5 -0.75 -0.94 1.0 0.14"

# Redirect output (stdout and stderr) to zebrafishMultiChannel.txt
exec > zebrafishMinChannel.txt 2>&1

# Loop over rendering modes 0 to 3
for m_mode in {0..3}; do
    # Loop over calls with and without -sh parameters
    for use_sh in true false; do
        # Set the -sh parameter if needed
        if [ "$use_sh" == true ]; then
            zebrafish_command_params="$zebrafish_light"
        else
            zebrafish_command_params=""
        fi

        for ((i = 0; i < ${#zebrafish_file_pairs[@]}; i++)); do
            file_pair=(${zebrafish_file_pairs[$i]})
            fr_files="${zebrafish_path}${file_pair[0]} "
            tf_files="${zebrafish_path}/numChanTF/${file_pair[1]} "
            
            outfile="zebrafish_min_${i}"
            
            if [ "$m_mode" == 3 ]; then
	    	macrocell_size="-mc 160 160 30"
	    else
	    	macrocell_size="-mc 80 80 15"
	    fi
            
            # Run the command with concatenated file pairs
            ../build/dTViewer -fr $fr_files \
               -t $tf_files \
               -cb 3.0 3.0 1.25 -r 1024 1024 -bg 0.3 0.3 0.3 $zebrafish_camera \
               $zebrafish_command_params -n 500 -wu 25 -m "$m_mode" -o "$outfile"\
	       $macrocell_size
        done
    done
done


# Loop over rendering modes 0 to 3
for m_mode in 7 8; do
    zebrafish_command_params=""

    for ((i = 0; i < ${#zebrafish_file_pairs[@]}; i++)); do
        file_pair=(${zebrafish_file_pairs[$i]})
        fr_files="${zebrafish_path}${file_pair[0]} "
        tf_files="${zebrafish_path}/numChanTF/${file_pair[1]} "
        
        outfile="zebrafish_min_${i}"
        
        if [ "$m_mode" == 3 ]; then
        macrocell_size="-mc 160 160 30"
    else
        macrocell_size="-mc 80 80 15"
    fi
        
        # Run the command with concatenated file pairs
        ../build/dTViewer -fr $fr_files \
            -t $tf_files \
            -cb 3.0 3.0 1.25 -r 1024 1024 -bg 0.3 0.3 0.3 $zebrafish_camera \
            $zebrafish_command_params -n 500 -wu 25 -m "$m_mode" -o "$outfile"\
        $macrocell_size
    done
done
