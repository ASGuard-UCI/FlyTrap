import cv2
import torch
import argparse
from tqdm import tqdm

from models.pysot.pysot.core.config import cfg
from models.pysot.pysot.models.model_builder import ModelBuilder
from models.pysot.pysot.tracker.tracker_builder import build_tracker


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def main(config_path, video_path, init_bbox, output_path):
    # load config
    cfg.merge_from_file(config_path)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # get total number of frames for progress bar
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    first_frame = True
    for frame in tqdm(get_frames(video_path), total=total_frames, desc="Processing frames"):
        if first_frame:
            tracker.init(frame, init_bbox)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = outputs['bbox']
            score = outputs['best_score']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                          (0, 255, 0), 2)
            text = f"Score: {score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(frame)

    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("snapshot", type=str, help="Path to the model snapshot")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("init_bbox", type=int, nargs=4, help="Initial bounding box (x, y, w, h)")
    parser.add_argument("--output_path", default='./output.mp4', type=str, help="Path to save the output video")
    args = parser.parse_args()

    main(args.config_path, args.video_path, tuple(args.init_bbox), args.output_path)