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


argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, help='configuration to specify the tracker')
argparser.add_argument('--patch', type=str, help='path to the adversarial patch')
argparser.add_argument('--video', type=str, help='specify the video to run the tracker, currently only support one video')
argparser.add_argument('--output', type=str, help='output path to save the results')
argparser.add_argument('--attack', action='store_true', help='whether to apply the attack')
args = argparser.parse_args()

class LSTM(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(4, 50, batch_first=True)
        self.fc = torch.nn.Linear(50, 5)
        self.softmax = torch.nn.Softmax(dim=2)
        self.trans_mat = torch.tensor([
            [.5, 0, -1, 0],
            [0, .5, 0, -1],
            [.5, 0, 1, 0],
            [0, .5, 0, 1]
        ]).float().to(device)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, 4)
        x = torch.matmul(x,self.trans_mat)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.softmax(x)


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
    # LSTM input
    fifo = []
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
                fifo = prepare_fifo(out['target_bbox'], fifo)
                
            if frame_idx > 10:
                # LSTM input
                x = torch.tensor(np.array(fifo), dtype=torch.float32).unsqueeze(0).cuda()
                pred_cls = class_map[alarmer(x).squeeze(0)[-1].argmax().item()]
                # visualize the result
                # if pred_cls == 'pedestrians': green, else red
                color = (0, 255, 0) if pred_cls == 'pedestrians' else (0, 0, 255)
                # convert [x1, y1, w, h] to [x1, y1, x2, y2] and clamp within H, W
                bbox = np.array(out['target_bbox'])
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox = np.clip(bbox, [0, 0, 0, 0], [1920-2, 1080-2, 1920-2, 1080-2])
                img = img.copy()
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), color, 2)
                img = cv2.putText(img, pred_cls, (int(bbox[0]), int(bbox[1]) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if data['apply_attack']:
                    count_attack += 1
                    if pred_cls != 'pedestrians':
                        alarm_attack += 1
                else:
                    count_benign += 1
                    if pred_cls != 'pedestrians':
                        alarm_benign += 1
            writer.write(img.astype(np.uint8))
    writer.release()
    
    print(f'Benign alarm rate: {alarm_benign/count_benign}')
    print(f'Attack alarm rate: {alarm_attack/count_attack}')


if __name__ == '__main__':
    
    args = argparser.parse_args()
    
    class_map = {
        0: 'bikes',
        1: 'buses',
        2: 'cars',
        3: 'pedestrians',
        4: 'trucks'
    }
    
    root = 'data/dataset_v4.0/eval'
    
    alarmer = LSTM()
    alarmer.load_state_dict(torch.load('ckpts/torch_bdd100k.pth')) # input shape [B, 10, 4]
    alarmer = alarmer.cuda()
    
    print(f'Evaluating video {args.video} with patch {args.patch}')
    
    cfg = mmengine.Config.fromfile(args.config)
    tracker = MODEL.build(cfg.tracker)
    applyer = APPLYER.build(cfg.applyer)
    renderer = APPLYER.build(cfg.renderer)
    test_dataloader = DATASETS.build(cfg.test_dataset)
    patch = torch.tensor(cv2.imread(args.patch)).permute(2, 0, 1).cuda()
    
    video_data = test_dataloader.videos[args.video]
    run(tracker, video_data, applyer, renderer, patch)

