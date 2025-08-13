# FlyTrap: Physical Distance-Pulling Attack Towards Camera-based Autonomous Target Tracking Systems

## About

Submit to NDSS 2026 Artifact Evaluation.

Autonomous Target Tracking (ATT) systems, especially ATT drones, are widely used in applications such as surveillance, border control, and law enforcement, while also being misused in stalking and destructive actions. Thus, the security of ATT is highly critical for real-world applications. Under the scope, we present a new type of attack: distance-pulling attacks (DPA) and a systematic study of it, which exploits vulnerabilities in ATT systems to dangerously reduce tracking distances, leading to drone capturing, increased susceptibility to sensor attacks, or even physical collisions. To achieve these goals, we present FlyTrap, a novel physical-world attack framework that employs an adversarial umbrella as a deployable and domain-specific attack vector. In this artifact, we provide the instructions to reproduce the main results in the paper to support our main claim and main contribution.

## Installation

Prepare the GCC version 9 and CUDA 11.3 environment.

```bash
# Install gcc version 9
sudo apt update
sudo apt install gcc-9 g++-9
# set environment variable
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
# verify gcc version 9
gcc --version
# install cuda 11.3 if not, using the following link:
# https://developer.nvidia.com/cuda-11.3.0-download-archive
export CUDA_HOME=/usr/local/cuda-11.3
```

Run the following command to create the environment; it takes around 10 minutes. The environment is successfully tested on Ubuntu 20.04 with CUDA 11.3.

```bash
# Create the environment
conda env create -f environment.yml
conda activate flytrap
# Install pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
# Install cython_bbox
pip install cython==0.27.3
pip install cython_bbox==0.1.3
# Install pysot
cd models/pysot && python setup.py build_ext --inplace
cd ../..
# Install MixFormer
cd models/MixFormer && python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
cd ../..
# Install VOGUES, this step might fail with the error: 
# SystemError: NULL result without error in PyObject_Call
# But you can ignore the error and continue.
cd models/VOGUES && python setup.py build develop
cd ../..
```

## Data Preparation

Download the dataset from [Google Drive](https://drive.google.com/file/d/1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W/view?usp=sharing). You can also download using `gdown` command:

```bash
mkdir data && cd data
pip install gdown
gdown 1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W
unzip dataset_v4.0.zip
cd ..
```

The dataset is organized as follows:

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

Then, download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1snpDOOAxZAvUrStP3QvJreaTe56gQhDV?usp=drive_link). You can also follow the following commands to download the checkpoints:

```
bash download.sh
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

## Reproduction

All the time estimation is based on a single NVIDIA RTX 3090 GPU. You can run each of the following experiments in parallel.

### Attack Evaluation (Tables I and II)

**Major Claims**: 

1. FlyTrap<sub>PDP</sub> can achieve better effectiveness than TGT. FlyTrap<sub>PDP</sub> can achieve better effectiveness than vanilla FlyTrap.
2. FlyTrap<sub>PDP</sub> can achieve better universality than TGT.

> The TGT baseline attack number might differ from what reported in the paper. We also update this part during the revision process. But the main claim that FlyTrap<sub>PDP</sub> can achieve better effectiveness than TGT is still valid.

#### FlyTrap Evaluation (~5 hours)

Run the following command to evaluate the effectiveness and universality of FlyTrap. All the trained adversarial patches are located in [`patches`](./patches).

```sh
# If this command takes too long
# please refer to the partial evaluation section below
bash scripts/eval_flytrap.sh
```

If the above command takes too long, you can also evaluate partial results by running a single model (e.g., `Mixformer`), this will take around 1 hour:

```sh
# FlyTrap w/o PDP (Estimated time: ~30 minutes)
python tools/main.py config/final/mixformer.py
# FlyTrap w/ PDP (Estimated time: ~30 minutes)
python tools/main.py config/final/mixformer_pdp.py
```

The results will be saved in `work_dirs/` with the name of the config file. There are three files in the `work_dirs/` directory:

- `benign_results_epoch-1.json`: The tracker output before the attack.
- `results_epoch-1.json`: The tracker output after the attack.
- `metric_epoch-1.json`: The mASR on each evaluation video.

To compute the detailed metric, we separate the evaluation videos into four categories based on the training set of the patch:

- **Effectiveness** to the same person and the same location.
- **Universality** to different persons and the same location (Universality to Person).
- **Universality** to the same person and different location (Universality to Location).
- **Universality** to different persons and different locations (Universality to Both).

To get the detailed metric, run the following command and follow the instructions in the terminal to get the final results and refer to Tables I and II in the paper. If you only evaluated partial results above, you can ignore the error raised by missing files for other models.

```bash
bash scripts/metric_summary.sh
```

#### TGT Evaluation (Full Evaluation, ~40 hours)

If the evaluation time takes too long, you may only want to evaluate partial results on one model; please refer to the [partial evaluation section](#tgt-evaluation-partial-evaluation-10-hours) below, which takes around 10 hours. If it's still too long, we provide the results json files for computing the detailed metric, please refer to the [detailed metric section](#detailed-metric) below.

Run the following command to evaluate the effectiveness and universality of TGT Images; all the TGT baseline attack patches are located in [`tgt`](./tgt):

```sh
bash scripts/eval_tgt.sh
```

The results will be saved in `work_dirs/` with the name of the config file. There are three files in the `work_dirs/` directory:

- `benign_results_epoch-1.json`: The tracker output before the attack.
- `results_epoch-1.json`: The tracker output after the attack.
- `metric_epoch-1.json`: The mASR on each evaluation video.

To get the detailed metric, run the following command to get the final results and refer to Tables I and II in the paper. You can specify the config file from the following list one by one and get the final results.

- `mixformer_tgt`
- `siamrpn_alex_tgt`
- `siamrpn_mob_tgt`
- `siamrpn_resnet_tgt`

```bash
python analysis/analyze_tgt_metric.py --input_dir work_dirs/<config_file_name>
```

#### TGT Evaluation (Partial Evaluation, ~10 hours)

You can specify the config file from the following list, depending on the victim model you want to evaluate.

- `config/final/mixformer_tgt.py`
- `config/final/siamrpn_alex_tgt.py`
- `config/final/siamrpn_mob_tgt.py`
- `config/final/siamrpn_resnet_tgt.py`

Then, run the following command to evaluate the effectiveness and universality of TGT Images on one model; all the TGT baseline attack patches are located in [`tgt`](./tgt). 

```bash
bash scripts/eval_tgt_partial.sh <config_file_name>
```

Then, for the final detailed metric, specify the config file name as the argument, including `mixformer_tgt`, `siamrpn_alex_tgt`, `siamrpn_mob_tgt`, `siamrpn_resnet_tgt`, depending on the victim model you evaluated above.

```sh
python analysis/analyze_tgt_metric.py --input_dir work_dirs/<config_file_name>/json_files/
```

#### TGT Metric Evaluation (Pre-computed Results, ~5 minute)

If the above command takes too long, you can also evaluate our generated results by running a single model (e.g., `Mixformer`), this will take around 1 minute:

```sh
# Download the results json files
gdown 1LCFybYCtAz2oCw4qMfOyCflHw6mqxWCN
unzip tgt_results.zip

# Evaluate the Mixformer results
python analysis/analyze_tgt_metric.py --input_dir tgt_results/mixformer_cvt_position_engine_final_tgt_scale15/json_files/
# Evaluate the SiamRPN-AlexNet results
python analysis/analyze_tgt_metric.py --input_dir tgt_results/siamrpn_alexnet_engine_final_tgt_scale15/json_files/
# Evaluate the SiamRPN-MobileNet results
python analysis/analyze_tgt_metric.py --input_dir tgt_results/siamrpn_mobilenet_engine_final_tgt_scale15/json_files/
# Evaluate the SiamRPN-ResNet results
python analysis/analyze_tgt_metric.py --input_dir tgt_results/siamrpn_resnet_engine_final_tgt_scale15/json_files/
```

### Defense Evaluation

**Major Claims**: 

1. FlyTrap<sub>ATG</sub> can reduce the true alarm rate compared to vanilla FlyTrap.

#### PercepGuard Evaluation (Table IV, ~2 hours)

Run the following command one by one to evaluate the benign case and the FlyTrap<sub>ATG</sub> attack case.

```sh
# Estimate time: ~1 hour
# Evaluation MixFormer
bash scripts/eval_perceguard.sh config/final/mixformer_percepguard.py patches/mixformer_flytrap_atg_percepguard.png
# Evaluation SiamRPN-ResNet
bash scripts/eval_perceguard.sh config/final/siamrpn_resnet_percepguard.py patches/siamrpn_resnet_flytrap_atg_percepguard.png
# Evaluation SiamRPN-Mobile
bash scripts/eval_perceguard.sh config/final/siamrpn_mob_percepguard.py patches/siamrpn_mobile_flytrap_atg_percepguard.png
# Evaluation SiamRPN-Alex
bash scripts/eval_perceguard.sh config/final/siamrpn_alex_percepguard.py patches/siamrpn_alex_flytrap_atg_percepguard.png
```

Run the following command one by one to evaluate the benign case and vanilla FlyTrap attack case.

```sh
# Estimate time: ~1 hour
# Evaluation MixFormer
bash scripts/eval_perceguard.sh config/final/mixformer_percepguard.py patches/mixformer_flytrap.png
# Evaluation SiamRPN-ResNet
bash scripts/eval_perceguard.sh config/final/siamrpn_resnet_percepguard.py patches/siamrpn_resnet_flytrap.png
# Evaluation SiamRPN-Mobile
bash scripts/eval_perceguard.sh config/final/siamrpn_mob_percepguard.py patches/siamrpn_mobile_flytrap.png
# Evaluation SiamRPN-Alex
bash scripts/eval_perceguard.sh config/final/siamrpn_alex_percepguard.py patches/siamrpn_alex_flytrap.png
```

#### VOGUES Evaluation (Table V, ~8 hours)

Run the following command one by one to evaluate the benign case and the FlyTrap<sub>ATG</sub> attack case. You can also want to evaluate a single model by specifying one of the following commands to save time.

```sh
# Estimate time: ~4 hours
# Evaluation MixFormer
bash scripts/eval_vogues.sh config/final/mixformer_vogues.py patches/mixformer_flytrap_atg_vogues.png
# Evaluation SiamRPN-ResNet
bash scripts/eval_vogues.sh config/final/siamrpn_resnet_vogues.py patches/siamrpn_resnet_flytrap_atg_vogues.png
# Evaluation SiamRPN-Mobile
bash scripts/eval_vogues.sh config/final/siamrpn_mob_vogues.py patches/siamrpn_mobile_flytrap_atg_vogues.png
# Evaluation SiamRPN-Alex
bash scripts/eval_vogues.sh config/final/siamrpn_alex_vogues.py patches/siamrpn_alex_flytrap_atg_vogues.png
```

Run the following command one by one to evaluate the benign case and vanilla FlyTrap attack case.

```sh
# Estimate time: ~4 hours
# Evaluation MixFormer
bash scripts/eval_vogues.sh config/final/mixformer_vogues.py patches/mixformer_flytrap.png
# Evaluation SiamRPN-ResNet
bash scripts/eval_vogues.sh config/final/siamrpn_resnet_vogues.py patches/siamrpn_resnet_flytrap.png
# Evaluation SiamRPN-Mobile
bash scripts/eval_vogues.sh config/final/siamrpn_mob_vogues.py patches/siamrpn_mobile_flytrap.png
# Evaluation SiamRPN-Alex
bash scripts/eval_vogues.sh config/final/siamrpn_alex_vogues.py patches/siamrpn_alex_flytrap.png
```

## Acknowledgments

We thank the following projects for their contributions:

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [PySOT](https://github.com/STVIR/pysot)
- [MixFormer](https://github.com/MCG-NJU/MixFormer)
- [PercepGuard](https://github.com/Harry1993/PercepGuard)