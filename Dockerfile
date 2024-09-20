# Base image with CUDA and cuDNN support
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge for Conda into /opt/miniforge3
RUN wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3 && \
    rm Miniforge3-Linux-x86_64.sh

# Set up environment variables for Conda
RUN echo "#!/bin/bash\n\
unset -f which\n\
source /opt/miniforge3/etc/profile.d/conda.sh\n\
export PATH=/opt/miniforge3/bin:\$PATH\n\
export PYTHONPATH=/opt/miniforge3/bin:\$PYTHONPATH" > /opt/env.sh

# Clone the gpudrive repository
RUN git clone --recursive https://github.com/Emerge-Lab/gpudrive.git

RUN git checkout ap_docker

# Set the working directory
WORKDIR /gpudrive

# Ensure Conda is available and create the environment
SHELL ["/bin/bash", "-c"]  # Use bash shell for running conda commands
RUN source /opt/env.sh && conda env create -f environment.yml

# Activate the environment and install project dependencies using Poetry
RUN echo "source /opt/env.sh && conda activate gpudrive" >> ~/.bashrc

ENV PYTHONPATH=/gpudrive:$PYTHONPATH

RUN source /opt/env.sh && conda activate gpudrive && poetry install

# Run tests
RUN source /opt/env.sh && conda activate gpudrive && pytest

# Automatically start in the /gpudrive directory and activate the conda environment
CMD ["bash", "-c", "source /opt/env.sh && conda activate gpudrive && cd /gpudrive && exec bash"]