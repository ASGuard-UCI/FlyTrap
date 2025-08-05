## This script contains code to visualize the results of the PerceptGuard defense
## The results is runned online, output final alarm rate and save the video with visualizations

## Under construction

import torch
import mmengine
import argparse
import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from mmengine.registry import DATASETS
from flytrap.builder import MODEL, APPLYER

from vogues.api import VOGUES

argparser = argparse.ArgumentParser()
argparser.add_argument('config', type=str, help='configuration to specify the tracker')
argparser.add_argument('patch', type=str, help='path to the adversarial patch')
argparser.add_argument('--video', type=str, default='person4_street1_instance2', help='specify the video to run the tracker, currently only support one video')
argparser.add_argument('--output', type=str, help='output path to save the results')
argparser.add_argument('--save', type=str, help='output path to save the json results')
argparser.add_argument('--attack', action='store_true', help='whether to apply the attack')
args = argparser.parse_args()


def prepare_fifo(data, fifo, max_len=10):
    # first convert [x1, y1, w, h] to [x1, y1, x2, y2]
    data = data.copy()
    data[2] += data[0]
    data[3] += data[1]
    # normalize to [0, 1]
    data = np.array(data) / np.array([1920, 1080, 1920, 1080])
    
    fifo.append(data)
    if len(fifo) > max_len:
        fifo.pop(0)
    return fifo


def run(tracker, video_dataset, applyer, renderer, patch):
   
    alarm_benign = 0
    count_benign = 0
    alarm_attack = 0
    count_attack = 0
    
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 24, (1920, 1080))
    render_patch = renderer(patch, train=False).detach().cpu()
    
    # For storing results to save as JSON
    results = []
    
    with torch.no_grad():
        for frame_idx, data in tqdm(enumerate(video_dataset), total=len(video_dataset), desc="Testing"):

            if frame_idx == 0:
                # initialize the tracker
                # [x1, y1, w, h]
                print(f'Initialize tracker at frame {frame_idx} with bbox {data["init_bbox"]}')
                img = data['img']
                tracker.initialize(img, {'init_bbox': data['init_bbox']})
            else:
                # track the target
                if data['apply_attack'] and args.attack:
                    img = torch.tensor(data['img'], dtype=torch.float32).unsqueeze(0)
                    coords = torch.tensor(data['coords']).unsqueeze(0)
                    # test_mode=True to fix patch orientation
                    img, _ = applyer(img, render_patch, coords, test_mode=True)
                    img = img.squeeze(0).numpy()
                else:
                    img = data['img']
                out = tracker.track(img) # [x1, y1, w, h]
                alarm = alarmer.validate(img, out['target_bbox'])
                info = alarmer.info()
                
                # Save frame results
                frame_result = {
                    'frame_idx': frame_idx,
                    'apply_attack': bool(data['apply_attack']),
                    'target_bbox': out['target_bbox'],
                    'shape': data['img'].shape,
                    'coords': data['coords'].tolist(),
                    'alarm': bool(alarm),
                    'info': info
                }
                results.append(frame_result)
                
                # Get the visualization with pose estimation
                img = alarmer.vis()
                
                # Draw IoU and LSTM score on top right
                text = f"IoU: {info['max_iou']:.3f}"
                text2 = f"LSTM: {info['lstm_score']:.3f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (0, 255, 0)  # Green color
                
                # Get text sizes
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)
                
                # Position text in top right with padding
                padding = 10
                x1 = img.shape[1] - text_width - padding
                x2 = img.shape[1] - text2_width - padding
                y1 = text_height + padding
                y2 = 2 * text_height + 2 * padding
                
                # Add text to image
                cv2.putText(img, text, (x1, y1), font, font_scale, color, thickness)
                cv2.putText(img, text2, (x2, y2), font, font_scale, color, thickness)
                
                # Draw tracking bbox
                track_bbox = np.array([out['target_bbox'][0], out['target_bbox'][1], 
                                        out['target_bbox'][0] + out['target_bbox'][2], 
                                        out['target_bbox'][1] + out['target_bbox'][3]])
                cv2.rectangle(img, 
                            (int(track_bbox[0]), int(track_bbox[1])), 
                            (int(track_bbox[2]), int(track_bbox[3])), 
                            (0, 0, 255),  # Blue color for track bbox
                            2)
                
                if data['apply_attack']:
                    count_attack += 1
                    if alarm:
                        alarm_attack += 1
                else:
                    count_benign += 1
                    if alarm:
                        alarm_benign += 1
            writer.write(img.astype(np.uint8))
    writer.release()
    
    # Save results to JSON if path is specified
    if args.save:
        json_output = {
            'video': args.video,
            'attack_enabled': bool(args.attack),
            'benign_alarm_rate': float(alarm_benign/count_benign if count_benign > 0 else 0),
            'attack_alarm_rate': float(alarm_attack/count_attack if count_attack > 0 else 0),
            'frames': results
        }
        if not os.path.exists(os.path.dirname(args.save)):
            os.makedirs(os.path.dirname(args.save))
        with open(args.save, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f'Results saved to {args.save}')
    
    print(f'Benign alarm rate: {alarm_benign/count_benign}')
    print(f'Attack alarm rate: {alarm_attack/count_attack}')


if __name__ == '__main__':
    
    root = 'data/dataset_v4.0/eval'
    
    cfg = 'models/VOGUES/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml'
    checkpoint_path = 'models/VOGUES/pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth'
    use_lstm = True
    lstm_checkpoint_path = 'models/VOGUES/model.pt'
    device = 'cuda'
    multi_obj = False
    iou_threshold = 0.5
    history_length = 20
    detector_name = 'yolo'
    
    alarmer = VOGUES(cfg, 
                     checkpoint_path, 
                     use_lstm, 
                     lstm_checkpoint_path, 
                     device, 
                     multi_obj, 
                     iou_threshold, 
                     history_length,
                     detector_name,
                     save_img=False,
                     vis=True,
                     showbox=True)
    
    print(f'Evaluating video {args.video} with patch {args.patch}')
    
    cfg = mmengine.Config.fromfile(args.config)
    tracker = MODEL.build(cfg.tracker)
    applyer = APPLYER.build(cfg.applyer)
    renderer = APPLYER.build(cfg.renderer)
    test_dataloader = DATASETS.build(cfg.test_dataset)
    patch = torch.tensor(cv2.imread(args.patch)).permute(2, 0, 1).cuda()
    
    video_data = test_dataloader.videos[args.video]
    run(tracker, video_data, applyer, renderer, patch)

