#!/bin/bash

#SBATCH --array=0-53%50
#SBATCH --job-name=paper_fig
#SBATCH --output=experiments/slurm/logs/output_%A_%a.txt
#SBATCH --error=experiments/slurm/logs/error_%A_%a.txt
#SBATCH --mem=10GB
#SBATCH --time=0-5:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1

SINGULARITY_IMAGE=hpc/nocturne.sif
OVERLAY_FILE=hpc/overlay-15GB-500K.ext3

singularity exec --nv --overlay /scratch/dc4971/gpudrive/hpc/overlay-15GB-500K.ext3:ro"     "/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"     /bin/bash baselines/scripts/bash_exec_paper_fig.sh "${SLURM_ARRAY_TASK_ID}"
echo "Successfully launched image."