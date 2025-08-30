FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda-11.3

# Fix NVIDIA repository GPG key issues and install system dependencies
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install Cython first (required for cython_bbox)
RUN pip install cython==0.29.3
# Install other requirements
RUN pip install -r requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

RUN apt-get update && apt-get install -y ninja-build

CMD ["/bin/bash"]