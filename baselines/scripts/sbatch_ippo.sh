#!/bin/bash

#SBATCH --array=0
#SBATCH --job-name=ippo
#SBATCH --output=baselines/ippo/logs/output_%A_%a.txt
#SBATCH --error=baselines/ippo/logs/error_%A_%a.txt
#SBATCH --mem=40GB
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1

singularity exec --nv --overlay /scratch/dc4971/gpudrive/hpc/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash baselines/scripts/bash_exec.sh
echo "Successfully launched image."
