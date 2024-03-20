#!/bin/bash

# FOR FIRSTN
# Loop from 1 to 30 to set the numEnvs value
for numEnvs in {1..150}; do
  echo "Running benchmarks for numEnvs = $numEnvs"

  MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs "$numEnvs"
done


#For RANDOMN
# echo "1..$numEnvsRange" | pv -p -s $numEnvsRange | while read numEnvs; do
#   echo "Running benchmarks for numEnvs = $numEnvs"
#   for i in $(seq 1 100); do
#     MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs "$numEnvs"
#   done
# done

#!/bin/bash

# Use seq to generate a range and pipe it through tqdm for the progress bar
# seq 1 150 | tqdm --total=150 | while read numEnvs; do
#   echo "Running benchmarks for numEnvs = $numEnvs"
#   for i in {1..100}; do
#     MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache python scripts/benchmark.py --numEnvs "$numEnvs"
#   done
# done
