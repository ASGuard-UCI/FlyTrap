## This script contains code to visualize the results of the VOGUES defense
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
argparser.add_argument('--videos', nargs='+', help='specify the videos to run the tracker, supports multiple videos')
argparser.add_argument('--output', type=str, help='output path to save the results')
argparser.add_argument('--attack', action='store_true', help='whether to apply the attack')


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


def run_single_video(tracker, video_dataset, applyer, renderer, patch, video_name, alarmer):
    """Run analysis on a single video and return alarm rates."""
    
    alarm_before = 0
    count_before = 0
    alarm_after = 0
    count_after = 0
    
    # Find the frame where attack/umbrella unfold starts
    attack_start_frame = None
    for frame_idx, data in enumerate(video_dataset):
        if data['apply_attack']:
            attack_start_frame = frame_idx
            break
    
    if attack_start_frame is None:
        print(f"Warning: No attack/umbrella unfold detected in video {video_name}")
        return None
    
    render_patch = renderer(patch, train=False).detach().cpu()
    
    # For storing results to save as JSON
    results = []
    
    with torch.no_grad():
        for frame_idx, data in tqdm(enumerate(video_dataset), total=len(video_dataset), desc=f"Testing {video_name}"):

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
                
                # Count alarms before and after attack/umbrella unfold
                if frame_idx < attack_start_frame:
                    count_before += 1
                    if alarm:
                        alarm_before += 1
                else:
                    count_after += 1
                    if alarm:
                        alarm_after += 1
    
    if count_before == 0 or count_after == 0:
        print(f"Warning: Insufficient frames for analysis in video {video_name}")
        return None
    
    alarm_rate_before = alarm_before / count_before
    alarm_rate_after = alarm_after / count_after
    
    print(f'{video_name} - Before: {alarm_rate_before:.4f} ({alarm_before}/{count_before})')
    print(f'{video_name} - After: {alarm_rate_after:.4f} ({alarm_after}/{count_after})')
    
    return {
        'video_name': video_name,
        'alarm_rate_before': alarm_rate_before,
        'alarm_rate_after': alarm_rate_after,
        'count_before': count_before,
        'count_after': count_after,
        'alarm_before': alarm_before,
        'alarm_after': alarm_after,
        'frames': results
    }


def run_multiple_videos(tracker, test_dataloader, applyer, renderer, patch, video_names, alarmer):
    """Run analysis on multiple videos and return aggregated results."""
    
    results = []
    
    for video_name in video_names:
        if video_name not in test_dataloader.videos:
            print(f"Warning: Video {video_name} not found in dataset")
            continue
            
        video_data = test_dataloader.videos[video_name]
        result = run_single_video(tracker, video_data, applyer, renderer, patch, video_name, alarmer)
        
        if result is not None:
            results.append(result)
    
    if not results:
        print("No valid results obtained")
        return None
    
    # Calculate averages
    avg_alarm_rate_before = np.mean([r['alarm_rate_before'] for r in results])
    avg_alarm_rate_after = np.mean([r['alarm_rate_after'] for r in results])
    
    # Prepare final results
    final_results = {
        'config': os.path.basename(args.config).replace('.py', ''),
        'patch': os.path.basename(args.patch),
        'attack': args.attack,
        'num_videos': len(results),
        'average_alarm_rate_before': float(avg_alarm_rate_before),
        'average_alarm_rate_after': float(avg_alarm_rate_after),
        'video_results': results
    }
    
    print(f"\nSummary:")
    print(f"Average alarm rate before: {avg_alarm_rate_before:.4f}")
    print(f"Average alarm rate after: {avg_alarm_rate_after:.4f}")
    
    return final_results


if __name__ == '__main__':
    
    args = argparser.parse_args()
    
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
    
    print(f'Evaluating videos {args.videos} with patch {args.patch}')
    
    cfg = mmengine.Config.fromfile(args.config)
    tracker = MODEL.build(cfg.tracker)
    applyer = APPLYER.build(cfg.applyer)
    renderer = APPLYER.build(cfg.renderer)
    test_dataloader = DATASETS.build(cfg.test_dataset)
    patch = torch.tensor(cv2.imread(args.patch)).permute(2, 0, 1).cuda()
    
    # Run analysis on multiple videos
    results = run_multiple_videos(tracker, test_dataloader, applyer, renderer, patch, args.videos, alarmer)
    
    if results is not None:
        # Save results to JSON file
        config_name = os.path.basename(args.config).replace('.py', '')
        patch_name = os.path.basename(args.patch).replace('.png', '').replace('.jpg', '')
        attack_suffix = 'attack' if args.attack else 'benign'
        
        # Create output directory
        output_dir = 'work_dirs/vogues_results'
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = os.path.join(output_dir, f"{config_name}_{patch_name}_{attack_suffix}.json")
        
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_filename}")

