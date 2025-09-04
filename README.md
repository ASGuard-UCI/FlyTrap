# ☂️ FlyTrap: Physical Distance-Pulling Attack Towards Camera-based Autonomous Target Tracking Systems

[![ArXiv](https://img.shields.io/badge/ArXiv-FAFAFA?logo=arxiv&logoColor=red&style=flat-square)]()
[![Demo](https://img.shields.io/badge/Demo-4285F4?logo=youtube&logoColor=white&style=flat-square)](https://sites.google.com/view/flytrap-adv)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white&style=flat-square)](https://drive.google.com/file/d/1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W/view)
[![Zenodo](https://img.shields.io/badge/Zenodo-FAFAFA?logo=zenodo&logoColor=blue&style=flat-square)](https://zenodo.org/records/16908024)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white&style=flat-square)](https://hub.docker.com/r/shaoyuanxie/flytrap)



## About

Autonomous Target Tracking (ATT) systems, especially ATT drones, are widely used in applications such as surveillance, border control, and law enforcement, while also being misused in stalking and destructive actions. Thus, the security of ATT is highly critical for real-world applications. Under the scope, we present a new type of attack: distance-pulling attacks (DPA) and a systematic study of it, which exploits vulnerabilities in ATT systems to dangerously reduce tracking distances, leading to drone capturing, increased susceptibility to sensor attacks, or even physical collisions. To achieve these goals, we present FlyTrap, a novel physical-world attack framework that employs an adversarial umbrella as a deployable and domain-specific attack vector. In this artifact, we provide the instructions to reproduce the main results in the paper to support our main claim and main contribution.

## Table of Contents

- [Installation](#installation)
- [Reproduction](#reproduction)
  - [Attack Evaluation](#1-attack-evaluation)
    - [FlyTrap Evaluation](#-11-flytrap-evaluation)
    - [TGT Evaluation](#-12-tgt-evaluation)
  - [Defense Evaluation](#2-defense-evaluation)
    - [PercepGuard Evaluation](#-21-percepguard-evaluation-table-iv-2-hours)
    - [VOGUES Evaluation](#-22-vogues-evaluation-table-v-8-hours)
- [Acknowledgments](#acknowledgments)



## Installation

Please refer to the [INSTALL.md](docs/INSTALL.md) for the installation instructions.

## Reproduction

All the time estimation is based on a single NVIDIA RTX 3090 GPU. You can run each of the following experiments in parallel.

### 1. Attack Evaluation

#### 🚀 1.1 FlyTrap Evaluation

We provide three options to evaluate the FlyTrap attack based on the available resources and time requirements.

- **Option 1: Full evaluation (~5 hours)**:

```sh
bash scripts/eval_flytrap.sh
```

- **Option 2: Partial evaluation (~1 hour)**:

You can select the following config files inside [`config/final`](./config/final):

| Model Name         | w/o PDP | w/ PDP |
|--------------------|:-------:|:------:|
| MixFormer          |   [`config/final/mixformer.py`](./config/final/mixformer.py)     |   [`config/final/mixformer_pdp.py`](./config/final/mixformer_pdp.py)    |
| SiamRPN-AlexNet    |   [`config/final/siamrpn_alex.py`](./config/final/siamrpn_alex.py)     |   [`config/final/siamrpn_alex_pdp.py`](./config/final/siamrpn_alex_pdp.py)    |
| SiamRPN-ResNet50   |   [`config/final/siamrpn_resnet.py`](./config/final/siamrpn_resnet.py)     |   [`config/final/siamrpn_resnet_pdp.py`](./config/final/siamrpn_resnet_pdp.py)    |
| SiamRPN-MobileV2   |   [`config/final/siamrpn_mob.py`](./config/final/siamrpn_mob.py)     |   [`config/final/siamrpn_mob_pdp.py`](./config/final/siamrpn_mob_pdp.py)    |


```sh
python tools/main.py <config_file>
```

- **Option 3: Pre-computed results (~10 minutes)**:

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white&style=flat-square)](https://drive.google.com/file/d/1JWd_yHl3pBgPSuHNr-XDkllULAm5TBDK/view?usp=drive_link) 

Download the pre-computed results from the above [link](https://drive.google.com/file/d/1JWd_yHl3pBgPSuHNr-XDkllULAm5TBDK/view?usp=drive_link). You can also download using the terminal command:
```bash
bash download/download_flytrap_results.sh
```

---

Finally, run the following command to compute the detailed metric:

- **Option 1 Full Evaluation**:

If you run the full evaluation above (**option 1**) or if you downloaded the pre-computed results (**option 3**), you can run the following command to compute the detailed metric.

```bash
bash scripts/metric_summary.sh
```

- **Option 2 Partial Evaluation**:

If you run the partial evaluation above (**option 2**), you can run the following command to compute the detailed metric. Please replace `<config>` with the config file name you evaluated above.

```bash
python analysis/analyze_result_metric.py --file work_dirs/<config>/json_files/results_epoch-1.json
```

#### 🚀 1.2 TGT Evaluation

We also provide three options to evaluate the TGT attack based on the available resources and time requirements.

- **Option 1: Full Evaluation (~40 hours)**:

Run the following command to evaluate the effectiveness and universality of TGT Images; all the TGT baseline attack patches are located in [`tgt`](./tgt):

```sh
bash scripts/eval_tgt.sh
```

- **Option 2: Partial Evaluation (~10 hours)**:

You can specify the config file from the following list, depending on the victim model you want to evaluate.

- `config/final/mixformer_tgt.py`
- `config/final/siamrpn_alex_tgt.py`
- `config/final/siamrpn_mob_tgt.py`
- `config/final/siamrpn_resnet_tgt.py`

```sh
bash scripts/eval_tgt_partial.sh <config_file>
```

- **Option 3: Pre-computed results (~10 minutes)**:

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white&style=flat-square)](https://drive.google.com/file/d/1JZkyJQy3-EsUkjAC0uxpTP3LQ-_38HQm/view?usp=sharing) 

Download the pre-computed results from the above [link](https://drive.google.com/file/d/1JZkyJQy3-EsUkjAC0uxpTP3LQ-_38HQm/view?usp=sharing). You can also download using the terminal command:
```bash
bash download/download_tgt_results.sh
```

---

If you run the full evaluation above (**option 1**) or if you downloaded the pre-computed results (**option 3**), you can run the following command to compute the detailed metric. If you run the partial evaluation above (**option 2**), please use the specific `<config_file>` you evaluated above.

```bash
python analysis/analyze_tgt_metric.py --input_dir work_dirs/mixformer_tgt/json_files
python analysis/analyze_tgt_metric.py --input_dir work_dirs/siamrpn_alex_tgt/json_files
python analysis/analyze_tgt_metric.py --input_dir work_dirs/siamrpn_mob_tgt/json_files
python analysis/analyze_tgt_metric.py --input_dir work_dirs/siamrpn_resnet_tgt/json_files
```

### 2. Defense Evaluation

We provide the code to evaluate the defense methods: USENIX'23 [`PercepGuard`](https://www.usenix.org/conference/usenixsecurity23/presentation/man) and USENIX'24 [`VOGUES`](https://www.usenix.org/conference/usenixsecurity24/presentation/muller).

#### 🚀 2.1 PercepGuard Evaluation (Table IV, ~2 hours)

Run the following command one by one to evaluate the benign case and the FlyTrap<sub>ATG</sub> attack case.

```sh

```

Run the following command one by one to evaluate the ATG and vanilla FlyTrap attack case. The results will be saved in `work_dirs/percepguard_results`. `before` means the false alarm rate and `after` means the true alarm rate after the attack.

```sh
# Evaluation MixFormer
bash scripts/eval_percepguard.sh config/final/mixformer_percepguard.py patches/mixformer_flytrap_atg_percepguard.png
bash scripts/eval_percepguard.sh config/final/mixformer_percepguard.py patches/mixformer_flytrap.png

# Evaluation SiamRPN-ResNet
bash scripts/eval_percepguard.sh config/final/siamrpn_resnet_percepguard.py patches/siamrpn_resnet_flytrap_atg_percepguard.png
bash scripts/eval_percepguard.sh config/final/siamrpn_resnet_percepguard.py patches/siamrpn_resnet_flytrap.png

# Evaluation SiamRPN-Mobile
bash scripts/eval_percepguard.sh config/final/siamrpn_mob_percepguard.py patches/siamrpn_mobile_flytrap_atg_percepguard.png
bash scripts/eval_percepguard.sh config/final/siamrpn_mob_percepguard.py patches/siamrpn_mobile_flytrap.png

# Evaluation SiamRPN-Alex
bash scripts/eval_percepguard.sh config/final/siamrpn_alex_percepguard.py patches/siamrpn_alex_flytrap_atg_percepguard.png
bash scripts/eval_percepguard.sh config/final/siamrpn_alex_percepguard.py patches/siamrpn_alex_flytrap.png
```

#### 🚀 2.2 VOGUES Evaluation (Table V, ~8 hours)

Run the following command one by one to evaluate the benign case and the FlyTrap<sub>ATG</sub> attack case. You can also want to evaluate a single model by specifying one of the following commands to save time.

```sh
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