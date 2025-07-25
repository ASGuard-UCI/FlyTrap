import argparse
import json
import os

import cv2
from tqdm import tqdm


def main(args):
    file = args.result_file
    video = args.video_file
    with open(file, 'r') as f:
        data = json.load(f)
    assert video in data.keys(), f"Video {video} should be in the result file {file}. Please check the video name. Available videos are {data.keys()}"
    assert video in os.listdir(
        args.root), f"Video {video} should be in the root directory {args.root}. Please check the video name."
    imgs = os.listdir(os.path.join(args.root, video))
    imgs = sorted([img for img in imgs if img.endswith('.jpg')])
    save_dir = os.path.join('./debug', video)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for data_item in tqdm(data[video]):
        frame_idx = data_item['frame_idx']
        img = cv2.imread(os.path.join(args.root, video, imgs[frame_idx]))
        gt_bbox = data_item['gt']['target_bbox']
        pred_bbox = data_item['out']['target_bbox']
        score = data_item['out']['target_score']
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])),
                      (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])),
                      (0, 255, 0), 2)
        cv2.rectangle(img, (int(pred_bbox[0]), int(pred_bbox[1])),
                      (int(pred_bbox[0] + pred_bbox[2]), int(pred_bbox[1] + pred_bbox[3])),
                      (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(gt_bbox[0]), int(gt_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, f'{frame_idx}.jpg'), img)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('result_file', type=str, help='Generated result file')
    argparse.add_argument('video_file', type=str, help='Video file to be visualized')
    argparse.add_argument('--root', type=str, help='Root to the video files', default='data/dataset_v4.0/eval')
    args = argparse.parse_args()
    main(args)
