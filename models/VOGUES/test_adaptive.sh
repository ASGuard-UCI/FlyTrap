python vogues/test_adaptive.py \
  --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml \
  --pose_checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth \
  --lstm_checkpoint model.pt \
  --video data/UCF-101/Biking/v_Biking_g01_c02.avi \
  --start_frame 30 \
  --x_offset 3.0 \
  --outdir output_vis \
  --mode shrink \
  --shrink_rate 0.1