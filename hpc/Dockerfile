FROM continuumio/miniconda3:24.1.2-0

# Set environment variables
ENV WANDB_CACHE_DIR=$WANDB_CACHE_DIR
ENV CONDA_ENVS_DIRS=$CONDA_ENVS_DIRS

# Install dependencies including cmake >=3.20
RUN apt-get update && apt-get install -y \
    software-properties-common \
    apt-transport-https \
    gpg-agent \
    wget \
    && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y cmake git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*