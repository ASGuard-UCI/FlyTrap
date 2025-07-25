import argparse
import os
from tqdm import tqdm
import cv2

import torch
from mmengine import Config
from mmengine.registry import DATASETS

import flytrap.builder as builder
import flytrap.utils as utils


def main(args):
    cfg = Config.fromfile(args.config)
    work_dir = 'work_dirs/' + os.path.basename(args.config).split('.')[0]
    patch_path = os.path.join(work_dir, 'patch.png')
    assert os.path.exists(patch_path), f"Patch path {patch_path} not exists!"
    patch = utils.load_patch(patch_path)

    renderer = builder.APPLYER.build(cfg.renderer)
    test_loader = DATASETS.build(cfg.test_dataset)
    applyer = builder.APPLYER.build(cfg.applyer)
    
    videos = getattr(test_loader, 'videos', None)
    assert videos is not None, "Test loader must have a 'videos' attribute."
    
    adv_videos_dir = os.path.join(work_dir, 'adv_videos')
    os.makedirs(adv_videos_dir, exist_ok=True)
    
    for video_name, video_dataset in videos.items():
        if video_name != args.video:
            continue
        render_patch = renderer(patch.cuda(), train=False).detach().cpu()
        video_path = os.path.join(adv_videos_dir, f"{video_name}.mp4")
        
        # Assuming all frames have the same shape
        first_frame = video_dataset[0]['img']
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Writing video to {video_path}")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        with torch.no_grad():
            for frame_idx, data in tqdm(enumerate(video_dataset), total=len(video_dataset), desc="Processing"):
                # if data['apply_attack']:
                #     img = torch.tensor(data['img'], dtype=torch.float32).unsqueeze(0)
                #     coords = torch.tensor(data['coords']).unsqueeze(0)
                #     img, _ = applyer(img, render_patch, coords, test_mode=True)
                #     img = img.squeeze(0).numpy()
                # else:
                img = data['img']
                img = img.astype('uint8')
                out.write(img)
        
        out.release()


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', default='config/mixformer/mixformer_cvt_position_engine_vogues_v5.py', type=str)
    argparse.add_argument('--video', default='person4_street1_instance2', type=str)
    args = argparse.parse_args()    
    main(args)