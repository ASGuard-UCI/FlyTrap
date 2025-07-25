# VOGUES: Validation of Object Guise using Estimated Components

Reproduction of USENIX Security 2024 paper: [VOGUES: Validation of Object Guise using Estimated Components](https://www.usenix.org/conference/usenixsecurity24/presentation/muller).

## Installation

Please follow the AlphaPose installation instructions [here](docs/INSTALL.md).

Possible bugs

```
pip install cython==0.27.3
pip install halpecocotools
# Then build the project
python setup.py build develop
```

## Dataset Preparation

Download UCF-101 dataset from the [website](https://www.crcv.ucf.edu/data/UCF101.php) and put it under `./data` folder.
```
.
└── UCF-101
    ├── ApplyEyeMakeup
    ├── ApplyLipstick
    ├── Archery
    ├── BabyCrawling
    ├── BalanceBeam
    ├── ...
```

## Get Started

### Pose Estimation

First generate the pose estimation results for UCF-101 dataset.

```bash
bash process_ucf101.sh
```

This will generate the pose estimation results for UCF-101 dataset and save them in `./data/ucf101_results` folder.

### Train the LSTM model

The idea is to train a LSTM model to classify the pose sequences as normal or anomalous.

```bash
bash train.sh
```

This will train the LSTM model and save the model in `./model.pt` file.

### Test the LSTM model

This will test the LSTM model on three types of sequences:
1. Corrupted sequences: replace one frame with random noise.
2. Random sequences: generate random sequences.
3. Benign sequences: the normal sequences from the training set.

```bash
bash test.sh --json_path output/ucf101_results/<json_file>
```

Also, we manually test the adaptive attacks without validation on the LSTM model to justify the reproducibility of the paper. In this attack, we manually move the detection bounding box away gradually and check if the LSTM can detect the anomaly.

```bash
bash test_adaptive.sh
```




### TODO

- [x] Add the VOGUES API.
- [ ] Add the evaluation metrics.
- [ ] Add the visualization of the results.
- [ ] Support VOGUES with Multiple Objects Tracking (MOT).
