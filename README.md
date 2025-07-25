# FlyTrap: Physical Distance-Pulling Attack Towards Camera-based Autonomous Target Tracking Systems

## About

Submit to NDSS 2026 Artifact Evaluation.

Autonomous Target Tracking (ATT) systems, especially ATT drones, are widely used in applications such as surveillance, border control, and law enforcement, while also being misused in stalking and destructive actions. Thus, the security of ATT is highly critical for real-world applications. Under the scope, we present a new type of attack: distance-pulling attacks (DPA) and a systematic study of it, which exploits vulnerabilities in ATT systems to dangerously reduce tracking distances, leading to drone capturing, increased susceptibility to sensor attacks, or even physical collisions. To achieve these goals, we present FlyTrap, a novel physical-world attack framework that employs an adversarial umbrella as a deployable and domain-specific attack vector. In this artifact, we provide the instructions to reproduce the main results in the paper to support our main claim and main contribution.

## Installation

Run the following command to create the environment:

```bash
conda env create -f environment.yml
conda activate flytrap
```

## Data Preparation

Download the dataset from [Google Drive](https://drive.google.com/file/d/1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W/view?usp=sharing). 

You can download using `gdown` command:

```bash
mkdir data && cd data
pip install gdown
gdown 1ezFU2-JiZC1szN5PnAUU_1ONDmAJM45W
unzip dataset_v4.0.zip
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


## Reproduction

### Effectiveness Evaluation (~1 hour)

> Corresponding to the effectiveness evaluation in **Table II** in the paper.

**Major Claims**: FlyTrap can achieve better effectiveness than TGT. FlyTrap<sub>PDP</sub> can achieve better effectiveness than vanilla FlyTrap.

#### TGT effectiveness.

Run the following command to evaluate target photo printing attack (TGT) effectiveness.

```sh
bash scripts/eval_tgt_effectiveness.sh
```

#### FlyTrap effectiveness.

Run the following command to evaluate FlyTrap effectiveness.

```sh
bash scripts/eval_flytrap_effectiveness.sh
```

#### FlyTrap<sub>PDP</sub> effectiveness.

Run the following command to evaluate FlyTrap with Progressive Distance Pulling (PDP) effectiveness.

```sh
bash scripts/eval_flytrap_pdp_effectiveness.sh
```

### Universality Evaluation (~1 hour)

> Corresponding to the universality evaluation in **Table III** in the paper.

#### TGT universality.

```sh
bash scripts/eval_tgt_universality.sh
```

#### FlyTrap universality.

```sh
bash scripts/eval_flytrap_universality.sh
```

### Defense Evaluation (~1 hour)

#### PercepGuard Evaluation.

> Corresponding to the PercepGuard evaluation in **Table IV** in the paper.

```sh
bash scripts/eval_perceguard.sh
```

#### VOGUES Evaluation.

> Corresponding to the VOGUES evaluation in **Table V** in the paper.

```sh
bash scripts/eval_vogues.sh
```

#### FlyTrap<sub>ATG</sub> Evaluation.

> Corresponding to the VOGUES evaluation in **Table VI** in the paper.