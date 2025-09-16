#!/bin/bash

# Define the paths
nyx_path=~/Desktop/Data/raw/nyx/

# Define the camera parameters
nyx_camera="-c -0.723570 1.378052 1.626585 -0.099032 0.852131 1.049213 0.386179 0.850533 -0.357014 45.000000"

# Define the file pairs arrays
nyx_file_pairs=(
    "baryon_512x512x512_float32.raw tfn_state_0.tf"
    "dark_matter_512x512x512_float32.raw tfn_state_1.tf"
    "temperature_512x512x512_float32.raw tfn_state_2.tf"
    "velmag_512x512x512_float32.raw tfn_state_3.tf"
    "velx_512x512x512_float32.raw tfn_state_4.tf"
    "vely_512x512x512_float32.raw tfn_state_5.tf"
    "velz_512x512x512_float32.raw tfn_state_6.tf"
    "baryon_512x512x512_float32.raw tfn_state_7.tf"
    "dark_matter_512x512x512_float32.raw tfn_state_8.tf"
    "temperature_512x512x512_float32.raw tfn_state_9.tf"
    "velmag_512x512x512_float32.raw tfn_state_10.tf"
    "velx_512x512x512_float32.raw tfn_state_11.tf"
)

# Define the light parameters
nyx_light="-sh 0.36 -0.5 0.15 1.0 0.15"

# Redirect output (stdout and stderr) to nyxMultiChannel.txt
exec > nyxMultiChannel.txt 2>&1

# Loop over rendering modes 2 and 3
for m_mode in {0..3}; do
    # Loop over calls with and without -sh parameters
    for use_sh in false true; do
        # Set the -sh parameter if needed
        if [ "$use_sh" == true ]; then
            nyx_command_params="$nyx_light"
        else
            nyx_command_params=""
        fi

        # Loop over the number of files added in each iteration
        # Form the concatenated file pairs based on the iteration
        concatenated_fr_files=""
        concatenated_tf_files=""
        for ((i = 0; i < ${#nyx_file_pairs[@]}; i+=2)); do
            file_pair=(${nyx_file_pairs[$i]})
            concatenated_fr_files+="${nyx_path}${file_pair[0]} "
            concatenated_tf_files+="${nyx_path}numChanTF/${file_pair[1]} "
            file_pair=(${nyx_file_pairs[$i+1]})
            concatenated_fr_files+="${nyx_path}${file_pair[0]} "
            concatenated_tf_files+="${nyx_path}numChanTF/${file_pair[1]} "
                
            
            ((n=i+2))
    	    outfile="nyx_$n"
    	    
    	    if [ "$m_mode" == 3 ]; then
	    	macrocell_size="-mc  256 256 256"
            else
                macrocell_size="-mc 128 128 128"
            fi

            # Run the command
            ../build/dTViewer -fr $concatenated_fr_files \
               -t $concatenated_tf_files \
                -bg 0.3 0.3 0.3 -r 1024 1024 $nyx_camera \
               $nyx_command_params -n 500 -wu 25 -m $m_mode -o $outfile\
	       $macrocell_size
        done
    done
done


# Loop over marching modes without shadows
for m_mode in 7 8; do
    # Loop over calls with and without -sh parameters
    nyx_command_params=""

    # Loop over the number of files added in each iteration
    # Form the concatenated file pairs based on the iteration
    concatenated_fr_files=""
    concatenated_tf_files=""
    for ((i = 0; i < ${#nyx_file_pairs[@]}; i+=2)); do
        file_pair=(${nyx_file_pairs[$i]})
        concatenated_fr_files+="${nyx_path}${file_pair[0]} "
        concatenated_tf_files+="${nyx_path}numChanTF/${file_pair[1]} "
        file_pair=(${nyx_file_pairs[$i+1]})
        concatenated_fr_files+="${nyx_path}${file_pair[0]} "
        concatenated_tf_files+="${nyx_path}numChanTF/${file_pair[1]} "
            
        
        ((n=i+2))
        outfile="nyx_$n"
        
        if [ "$m_mode" == 3 ]; then
        macrocell_size="-mc  256 256 256"
        else
            macrocell_size="-mc 128 128 128"
        fi

        # Run the command
        ../build/dTViewer -fr $concatenated_fr_files \
            -t $concatenated_tf_files \
            -bg 0.3 0.3 0.3 -r 1024 1024 $nyx_camera \
            $nyx_command_params -n 500 -wu 25 -m $m_mode -o $outfile\
        $macrocell_size
    done
done
