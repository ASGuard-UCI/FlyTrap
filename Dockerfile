FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essentials, zsh, gcc-9/g++-9, and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    zsh \
    build-essential \
    gcc-9 g++-9 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set GCC/G++ 9 as default
ENV CC=/usr/bin/gcc-9
ENV CXX=/usr/bin/g++-9

# Install CUDA 11.3
RUN wget --no-verbose https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run && \
    chmod +x cuda_11.3.0_465.19.01_linux.run && \
    ./cuda_11.3.0_465.19.01_linux.run --silent --toolkit > /tmp/cuda_install.log 2>&1 && \
    rm cuda_11.3.0_465.19.01_linux.run

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install Miniconda (latest for Linux x86_64)
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Ensure conda is in PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda for zsh
RUN $CONDA_DIR/bin/conda init zsh

# Add environment variables to .zshrc
RUN echo 'export CC=/usr/bin/gcc-9' >> ~/.zshrc && \
    echo 'export CXX=/usr/bin/g++-9' >> ~/.zshrc && \
    echo 'export CUDA_HOME=/usr/local/cuda-11.3' >> ~/.zshrc && \
    echo 'export PATH=${CUDA_HOME}/bin:$PATH' >> ~/.zshrc && \
    echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}' >> ~/.zshrc && \
    echo 'export CUDA_VISIBLE_DEVICES="0,1"' >> ~/.zshrc


# Set zsh as default shell
SHELL ["/usr/bin/zsh", "-c"]
ENV SHELL=/usr/bin/zsh

WORKDIR /workspace
CMD ["zsh"]