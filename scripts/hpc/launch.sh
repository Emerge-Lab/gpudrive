# Script for launching the singularity image for gpudrive
# ---
# Note: Script must be run from the gpudrive repository (i.e. /scratch/$USER/gpudrive/)
# and executed each time you want to use gpudrive (lab version) on the HPC (e.g. Greene).

# Constants
PROJECT="gpudrive"
PROJECT_DOCKER=docker://daphnecor/gpudrive
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif 
OVERLAY_LOC=hpc/overlay-15GB-500K.ext3
OVERLAY_FILE=overlay-15GB-500K.ext3

# Overwrite wandb cache dir to avoid storage capacity problems
WANDB_CACHE_DIR='${WANDB_CACHE_DIR}'

# Check if singularity image exists, if not pull Singularity image from Docker Hub
if [ ! -f "${SINGULARITY_IMAGE}" ]; then
    echo "Pulling Docker container from ${PROJECT_DOCKER}"
    singularity pull $SINGULARITY_IMAGE $PROJECT_DOCKER
fi

# Check if overlay file exists, if not create it
if [ ! -f "./hpc/${OVERLAY_FILE}" ]; then  # Overlay file does not exist
    echo "Setting up ${PROJECT_DOCKER} with initial overlay ${OVERLAY_FILE}.gz"

    if [ ! -f "${OVERLAY_FILE}.gz" ]; then  # Overlay file has not been copied yet
        echo "Copying overlay ${OVERLAY_FILE}.gz from ${OVERLAY_LOC}..."
        cp -rp "${OVERLAY_LOC}/${OVERLAY_FILE}.gz" ./hpc -n
        echo "Unzipping overlay ./hpc/${OVERLAY_FILE}.gz..."
        gunzip "./hpc/${OVERLAY_FILE}.gz" -n
    fi

    # Launch singularity for the first time
    echo 'Launching singularity image in WRITE (edit) mode...'

    # Welcome message
    echo "Run the following to initialize ${PROJECT}:"
    echo "  (1) create conda environment: 'conda env create -f environment.yml'"
    echo "  (2) activate conda environment: 'conda activate gpudrive'"
    echo "  (3) install gpudrive: 'poetry install'"

    # Launch singularity image in write mode
    singularity exec --nv --overlay "./hpc/${OVERLAY_FILE}:rw" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash

else  # Overlay Singularity image and overlay file exist

    # Launch singularity
    echo 'Launching singularity image in OVERLAY (use) mode...'

    # Welcome message
    echo "Run the following to activate the Python environment:"
    echo "  (1) activate conda environment: 'conda activate gpudrive'"

    # Launch singularity image in use mode
    singularity exec --nv --overlay "./hpc/${OVERLAY_FILE}:ro" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash

fi
