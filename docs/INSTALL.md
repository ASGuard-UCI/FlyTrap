# Installation

## Table of Contents
- [Requirements](#requirements)
    - [Conda Installation](#conda-installation)
    - [Docker Installation](#docker-installation)
- [Data Preparation](#data-preparation)
    - [Google Drive](#google-drive)
    - [Zenodo](#zenodo)
- [Model Preparation](#model-preparation)

## Requirements

We support two installation methods:

1. [Conda environment](#conda-installation)
2. [Docker container](#docker-installation)

We recommend using the Docker container for a consistent and reproducible installation environment.

### Conda Installation

Please ensure that you have `CUDA 11.3` installed on your system. Then run the following command to create the environment:

```bash
# Create new environment
conda create -n flytrap python=3.8
conda activate flytrap

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-11.3

# Install dependencies
pip install cython==0.29.3
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# Install models
bash install.sh
```

### Docker Installation

We provide a `Dockerfile` in the root directory. You can also pull our pre-built docker image from [Docker Hub]().

- Option 1: Build from Dockerfile

```bash
docker build -t flytrap .
```

- Option 2: Pull from Docker Hub

```bash
docker pull shaoyuanxie/flytrap:latest
```

After preparing the environment, you can run the following command to start the container:

```bash
bash run_docker.sh
```

Then, inside the container, you can run the following command to build the models:

```bash
cd /workspace/flytrap
bash install.sh
```

## Data Preparation

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white&style=flat-square)](https://drive.google.com/file/d/1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W/view) 
[![Zenodo](https://img.shields.io/badge/Zenodo-FAFAFA?logo=zenodo&logoColor=blue&style=flat-square)](https://zenodo.org/records/16908024)

We provide two ways to prepare the data.
1. Google Drive

Please download the dataset from the [link](https://drive.google.com/file/d/1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W/view) above. You can also download using the terminal command:
```bash
bash download_data.sh
```

2. Zenodo

Please download the dataset from the [link](https://zenodo.org/records/16908024) above.

---

Finally, the dataset is organized as follows:
```
├── data
│   └── dataset_v4.0
│       ├── annotations
│       │   ├── eval
│       │   ├── train_search
│       │   └── train_template
│       ├── collect_eval.json
│       ├── collect_train_final.json
│       ├── collect_train.json
│       ├── eval
│       ├── README.md
│       ├── scripts
│       ├── TGT
│       ├── train_search
│       └── train_template
```

## Model Preparation

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white&style=flat-square)](https://drive.google.com/drive/u/2/folders/1snpDOOAxZAvUrStP3QvJreaTe56gQhDV) 

You can download the checkpoints from the [link](https://drive.google.com/drive/u/2/folders/1snpDOOAxZAvUrStP3QvJreaTe56gQhDV) above. You can also download using the terminal command:

```bash
bash download_model.sh
```

The checkpoints are organized as follows: if you used the above commands to download the checkpoints, they're already in the correct place.
```
├── ckpts/
│   ├── torch_bdd100k.pth
│   ├── siamrpn_r50_l234_dwxcorr.pth
│   ├── siamrpn_mobilev2_l234_dwxcorr.pth
│   └── siamrpn_alex_dwxcorr.pth
├── models/
│   ├── MixFormer/
│   │   └── models/
│   │       └── mixformer_online_22k.pth.tar
│   └── VOGUES/
│       └── model.pt
│       └── pretrained_models/
│           └── multi_domain_fast50_dcn_combined_256x192.pth
```