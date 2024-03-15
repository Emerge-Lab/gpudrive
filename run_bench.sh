#!/bin/bash

# Loop from 1 to 30 to set the numEnvs value
for numEnvs in {1..30}; do
  echo "Running benchmarks for numEnvs = $numEnvs"
  
  # Inner loop to run the command 20 times for each numEnvs value
  for run in {1..20}; do
    echo "Iteration $run for numEnvs = $numEnvs"
    python scripts/benchmark.py --numEnvs "$numEnvs"
  done
  
done
