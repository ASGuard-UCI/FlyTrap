#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

python tools/demo_video.py \
    ./models/ATCover/models/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    ./models/ATCover/ckpts/siamrpn_r50_l234_dwxcorr.pth \
    /mnt/nas/shaoyuan/drone/phy_adv_data/DJI_20241207163027_0002_D.MP4 \
    837 571 99 324 \
    --output_path ./output.mp4