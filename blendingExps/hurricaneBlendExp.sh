#!/bin/bash

# Define the paths
hurricane_path=~/Desktop/Data/raw/hurricane/

# Define the camera parameters
hurricane_camera="-c 144.862579 172.479889 388.386383 144.862579 172.479889 387.386383 0.000000 1.000000 0.000000 45.000000"

# Define the file pairs arrays
hurricane_file_pairs=(
    "QCLOUDf48log10_500x500x100_float32.raw tfn_state_0.tf"
    "QRAINf48log10_500x500x100_float32.raw tfn_state_1.tf"
    "QSNOWf48log10_500x500x100_float32.raw tfn_state_2.tf"
    "QVAPORf48_500x500x100_float32.raw tfn_state_3.tf"
)

# Redirect output (stdout and stderr) to hurricaneRenderModes.txt
exec > hurricaneBlendExps.txt 2>&1

for ((i = 0; i < ${#hurricane_file_pairs[@]}; i++)); do
	file_pair=(${hurricane_file_pairs[$i]})
        fr_params+="${hurricane_path}${file_pair[0]} "
        t_params+="${hurricane_path}/blendingTF/${file_pair[1]} "
done
# Loop over specified render modes
for m_mode in 2 4 5 6 8 9; do

    # Run the command
    ../build/dTViewer -fr $fr_params -t $t_params \
       -cb -bg 0.0 0.0 0.0 -r 1024 1024 $hurricane_camera -m "$m_mode" -n 500 -dt 0.045 -mc 125 125 25 -o hurricane
done

