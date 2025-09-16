#!/bin/bash

# Define the paths
miranda_path=~/Desktop/Data/raw/miranda/

# Define the camera parameters
miranda_camera="-c 577.905273 231.364685 -155.170486 577.162842 231.294373 -154.504288 -0.667723 -0.002370 -0.744406 45.000000"

# Define the file pairs arrays
miranda_file_pairs=(
    "density_384x384x256_double64.raw tfn_state_0.tf"
    "velmag_384x384x256_double64.raw tfn_state_1.tf"
    "diffusivity_384x384x256_double64.raw tfn_state_2.tf"
    "pressure_384x384x256_double64.raw tfn_state_3.tf"
    "viscocity_384x384x256_double64.raw tfn_state_4.tf"
    "velx_384x384x256_double64.raw tfn_state_5.tf"
    "vely_384x384x256_double64.raw tfn_state_6.tf"
    "velz_384x384x256_double64.raw tfn_state_7.tf"
    "density_384x384x256_double64.raw tfn_state_8.tf"
    "diffusivity_384x384x256_double64.raw tfn_state_9.tf"
    "velmag_384x384x256_double64.raw tfn_state_10.tf"
    "pressure_384x384x256_double64.raw tfn_state_11.tf"
)

# Define the light parameters
miranda_light="-sh -0.69 -0.89 0.55 1.0 0.1"

# Redirect output (stdout and stderr) to mirandaMultiChannel.txt
exec > mirandaMultiChannel.txt 2>&1

# Loop over rendering modes 0 to 7
for m_mode in {0..3}; do
    # Loop over calls with and without -sh parameters
    for use_sh in true false; do
        # Set the -sh parameter if needed
        if [ "$use_sh" == true ]; then
            miranda_command_params="$miranda_light"
        else
            miranda_command_params=""
        fi

        # Loop over the number of files added in each iteration
        # Form the concatenated file pairs based on the iteration
        concatenated_fr_files=""
        concatenated_tf_files=""
        for ((i = 0; i < ${#miranda_file_pairs[@]}; i+=2)); do
            file_pair=(${miranda_file_pairs[$i]})
            concatenated_fr_files+="${miranda_path}${file_pair[0]} "
            concatenated_tf_files+="${miranda_path}numChanTF/${file_pair[1]} "
            file_pair=(${miranda_file_pairs[$i+1]})
            concatenated_fr_files+="${miranda_path}${file_pair[0]} "
            concatenated_tf_files+="${miranda_path}numChanTF/${file_pair[1]} "
                
            
            ((n=i+2))
    	    outfile="miranda_$n"
    	    
    	    if [ "$m_mode" == 3 ]; then
	    	macrocell_size="-mc 192 192 128"
	    else
	    	macrocell_size="-mc 96 96 64"
	    fi

            # Run the command
            ../build/dTViewer -fr $concatenated_fr_files \
               -t $concatenated_tf_files \
               -cb -bg 0.3 0.3 0.3 -r 1024 1024 $miranda_camera \
               $miranda_command_params -dt 0.25 -n 500 -wu 25 -m $m_mode -o $outfile\
	       $macrocell_size
        done
    done
done

# Loop over rendering modes 0 to 7
for m_mode in 7 8; do
    # Loop over calls with and without -sh parameters
    miranda_command_params=""
    # Loop over the number of files added in each iteration
    # Form the concatenated file pairs based on the iteration
    concatenated_fr_files=""
    concatenated_tf_files=""
    for ((i = 0; i < ${#miranda_file_pairs[@]}; i+=2)); do
        file_pair=(${miranda_file_pairs[$i]})
        concatenated_fr_files+="${miranda_path}${file_pair[0]} "
        concatenated_tf_files+="${miranda_path}numChanTF/${file_pair[1]} "
        file_pair=(${miranda_file_pairs[$i+1]})
        concatenated_fr_files+="${miranda_path}${file_pair[0]} "
        concatenated_tf_files+="${miranda_path}numChanTF/${file_pair[1]} "
            
        
        ((n=i+2))
        outfile="miranda_$n"
        
        if [ "$m_mode" == 3 ]; then
        macrocell_size="-mc 192 192 128"
    else
        macrocell_size="-mc 96 96 64"
    fi

        # Run the command
        ../build/dTViewer -fr $concatenated_fr_files \
            -t $concatenated_tf_files \
            -cb -bg 0.3 0.3 0.3 -r 1024 1024 $miranda_camera \
            $miranda_command_params -dt 0.25 -n 500 -wu 25 -m $m_mode -o $outfile\
        $macrocell_size
    done
done

