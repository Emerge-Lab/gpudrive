# Base image with CUDA and cuDNN support                                                                                      
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y -q --no-install-recommends \
	software-properties-common \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        wget \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        mesa-common-dev \
        libc++1 \
        openssh-client \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh

# Copy the gpudrive repository
COPY . /gpudrive
WORKDIR /gpudrive
RUN git submodule update --init --recursive

# Install python part using uv
RUN uv sync --frozen

ENV MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache

RUN mkdir build
WORKDIR /gpudrive/build
RUN uv run cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && find external -type f -name "*.tar" -delete
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH uv run make -j
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
WORKDIR /gpudrive

CMD ["/bin/bash"]
LABEL org.opencontainers.image.source=https://github.com/Emerge-Lab/gpudrive
