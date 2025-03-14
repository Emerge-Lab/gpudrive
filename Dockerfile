# Base image with CUDA and cuDNN support                                                                                      
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
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
        openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN apt-add-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 11

RUN apt-get remove -y cmake && pip3 install --upgrade cmake

# Clone the gpudrive repository
RUN git clone --recursive https://github.com/Emerge-Lab/gpudrive.git --branch dev-packaged
ENV MADRONA_MWGPU_KERNEL_CACHE=./gpudrive_cache

WORKDIR /gpudrive
RUN mkdir build
WORKDIR /gpudrive/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH make -j
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
WORKDIR /gpudrive

RUN pip3 install -e .[pufferlib]

CMD ["/bin/bash"]
LABEL org.opencontainers.image.source=https://github.com/Emerge-Lab/gpudrive
