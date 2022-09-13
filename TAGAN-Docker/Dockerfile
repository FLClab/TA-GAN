FROM nvidia/cuda:10.0-base-ubuntu16.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -L -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install dominate
RUN pip install dominate==2.3.1

# Install tifffile
RUN pip install tifffile

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -y

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

RUN pip install --upgrade --force-reinstall pip==9.0.3

# Install tifffile
RUN pip install tifffile --disable-pip-version-check

# Install scikit-image and scikit-learn
RUN pip install scikit-image --disable-pip-version-check
RUN pip install scikit-learn --disable-pip-version-check

# Install pandas
RUN pip install pandas

# Install tqdm
RUN pip install tqdm
