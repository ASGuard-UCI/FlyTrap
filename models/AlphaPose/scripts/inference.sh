python ./scripts/custom_demo_inference_temporal_analyze.py \
    --cfg ./configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml \
    --checkpoint ./pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth \
    --video /home/shaoyux/models/ATCover/work_dirs/mixformer_cvt_position_engine_vogues_v5/adv_videos/person4_street1_instance2.mp4 \
    --outdir ./examples/res \
    --detector yolo \
    --showbox \
    --save_img \
    --save_video
    # --list im_names.txt \
    # --video /home/shaoyux/models/ATCover/work_dirs/mixformer_cvt_position_engine_vogues_v3/adv_videos/person4_street1_instance2.mp4 \
