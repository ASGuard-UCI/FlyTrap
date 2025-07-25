CONFIG=configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CHECKPOINT=pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth
VIDEO_NAME=data/UCF-101/BasketballDunk/v_BasketballDunk_g01_c02.avi
OUTPUT_DIR=output_vis/


bash ./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME} ${OUTPUT_DIR}