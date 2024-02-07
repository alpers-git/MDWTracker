#!/bin/bash

# Define the paths
hurricane_path=~/Desktop/Data/raw/hurricane/

# Define the camera parameters
hurricane_camera="-c 251.565598 250.310257 716.027161 251.565598 250.310257 715.027161 0.000000 1.000000 0.000000 45.000000"

# Define the file pairs arrays
hurricane_file_pairs=(
    "CLOUDf48log10_500x500x100_float32.raw tfn_state_0.tf"
    "QCLOUDf48log10_500x500x100_float32.raw tfn_state_1.tf"
    "PRECIPf48log10_500x500x100_float32.raw tfn_state_2.tf"
    "QGRAUPf48log10_500x500x100_float32.raw tfn_state_3.tf"
    "QICEf48log10_500x500x100_float32.raw tfn_state_4.tf"
    "QRAINf48log10_500x500x100_float32.raw tfn_state_5.tf"
    "QSNOWf48log10_500x500x100_float32.raw tfn_state_6.tf"
    "QVAPORf48_500x500x100_float32.raw tfn_state_7.tf"
    "TCf48_500x500x100_float32.raw tfn_state_8.tf"
    "Uf48_500x500x100_float32.raw tfn_state_9.tf"
    "Vf48_500x500x100_float32.raw tfn_state_10.tf"
    "Wf48_500x500x100_float32.raw tfn_state_11.tf"
)

# Define the light parameters
hurricane_light="-sh 0.09 -1.0 -1.0 1.0 0.1"

# Redirect output (stdout and stderr) to hurricaneMultiChannel.txt
exec > hurricaneMinChannel.txt 2>&1

# Loop over rendering modes 0 to 7
for m_mode in {0..3}; do
    # Loop over calls with and without -sh parameters
    for use_sh in true false; do
        # Set the -sh parameter if needed
        if [ "$use_sh" == true ]; then
            hurricane_command_params="$hurricane_light"
        else
            hurricane_command_params=""
        fi

        # Loop over the number of files added in each iteration
        # Form the concatenated file pairs based on the iteration
        for ((i = 0; i < ${#hurricane_file_pairs[@]}; i++)); do
            file_pair=(${hurricane_file_pairs[$i]})
            fr_files="${hurricane_path}${file_pair[0]} "
            tf_files="${hurricane_path}numChanTF/${file_pair[1]} "
                
            
    	    outfile="hurricane_min_$i"
    	    
    	    if [ "$m_mode" == 3 ]; then
	    	macrocell_size="-mc 250 250 50"
	    else
	    	macrocell_size="-mc 125 125 25"
	    fi

            # Run the command
            ../build/dTViewer -fr $fr_files \
               -t $tf_files \
               -cb -bg 0.3 0.3 0.3 -r 1024 1024 $hurricane_camera \
               $hurricane_command_params -n 500 -wu 25 -dt 0.085 -m $m_mode -o $outfile\
	       $macrocell_size
        done
    done
done


# Loop over rendering modes 7 and 8
for m_mode in 7 8; do
    
    hurricane_command_params=""

    # Loop over the number of files added in each iteration
    # Form the concatenated file pairs based on the iteration
    for ((i = 0; i < ${#hurricane_file_pairs[@]}; i++)); do
        file_pair=(${hurricane_file_pairs[$i]})
        fr_files="${hurricane_path}${file_pair[0]} "
        tf_files="${hurricane_path}numChanTF/${file_pair[1]} "
            
        
        outfile="hurricane_min_$i"
        
        if [ "$m_mode" == 3 ]; then
        macrocell_size="-mc 250 250 50"
    else
        macrocell_size="-mc 125 125 25"
    fi

        # Run the command
        ../build/dTViewer -fr $fr_files \
            -t $tf_files \
            -cb -bg 0.3 0.3 0.3 -r 1024 1024 $hurricane_camera \
            $hurricane_command_params -n 500 -wu 25 -dt 0.085 -m $m_mode -o $outfile\
        $macrocell_size
    done
done