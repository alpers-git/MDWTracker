#!/bin/bash

../build/dTViewer -fr ~/Desktop/Dev/dTracker/blendingExps/posX_500x500x500_uint8.raw -t ~/Desktop/Dev/dTracker/blendingExps/tfn_state_0.tf -c 0.500000 0.500000 2.272339 0.500000 0.500000 1.272339 0.000000 1.000000 0.000000 45.000000 -m 2 -n 500 -o red

../build/dTViewer -fr ~/Desktop/Dev/dTracker/blendingExps/negX_500x500x500_uint8.raw -t ~/Desktop/Dev/dTracker/blendingExps/tfn_state_1.tf -c 0.500000 0.500000 2.272339 0.500000 0.500000 1.272339 0.000000 1.000000 0.000000 45.000000 -m 2 -n 500 -o green

../build/dTViewer -fr ~/Desktop/Dev/dTracker/blendingExps/negY_500x500x500_uint8.raw -t ~/Desktop/Dev/dTracker/blendingExps/tfn_state_2.tf -c 0.500000 0.500000 2.272339 0.500000 0.500000 1.272339 0.000000 1.000000 0.000000 45.000000 -m 2 -n 500 -o blue

for RENDER_MODE in {0..9}; do
  ../build/dTViewer -fr ~/Desktop/Dev/dTracker/blendingExps/posX_500x500x500_uint8.raw ~/Desktop/Dev/dTracker/blendingExps/negX_500x500x500_uint8.raw ~/Desktop/Dev/dTracker/blendingExps/negY_500x500x500_uint8.raw -t ~/Desktop/Dev/dTracker/blendingExps/tfn_state_0.tf ~/Desktop/Dev/dTracker/blendingExps/tfn_state_1.tf ~/Desktop/Dev/dTracker/blendingExps/tfn_state_2.tf -c 0.500000 0.500000 2.272339 0.500000 0.500000 1.272339 0.000000 1.000000 0.000000 45.000000 -m $RENDER_MODE -n 500 -o basic
done
