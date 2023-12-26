#!/bin/bash

# File to store the output
output_file="sim_results.txt"

# Clear the output file
> $output_file

# Loop from 1 to 120
for i in {1..120}
do
    echo "Running simulation with $i parallel sims"
    # Run your program and capture the output
    output=$(./build/headless CUDA $i 90 | grep 'FPS')
    
    # Extract the FPS value
    fps=$(echo $output | awk '{print $2}')
    
    # Write the result to the file
    echo "Sim $i: $fps FPS" >> $output_file
done

echo "All simulations completed. Results stored in $output_file."
