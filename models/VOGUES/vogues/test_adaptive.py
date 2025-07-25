# This script is used to test the reproduction of LSTM

import torch
import argparse
import json
import os
import cv2
import numpy as np
import time
from vogues.model import OneClassLSTM
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime

class DetectionInjector:
    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector
        self.last_valid_box = None
        self.x_offset = opt.x_offset
        self.current_offset = 0
        self.mode = opt.mode
        self.shrink_rate = opt.shrink_rate
        self.original_size = None

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma,
            train=False, add_dpg=False, gpu_device=self.device)

    def process(self, im_name, image, frame_idx, start_frame):
        # Pre-process image
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image
        im_dim = orig_img.shape[1], orig_img.shape[0]
        im_name = os.path.basename(im_name) if isinstance(im_name, str) else im_name
        
        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        # If before start_frame, use normal detection
        if frame_idx < start_frame:
            with torch.no_grad():
                dets = self.detector.images_detection(img, im_dim)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    return None, orig_img, im_name, None, None, None, None
                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                dets = dets.cpu()
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]
                ids = torch.zeros(scores.shape)

            boxes = boxes[dets[:, 0] == 0]
            if isinstance(boxes, int) or boxes.shape[0] == 0:
                return None, orig_img, im_name, None, None, None, None
            
            # Save the last valid box for injection
            self.last_valid_box = boxes[0].clone()
            self.original_size = self.last_valid_box.clone()
            self.current_offset = 0
        else:
            # Inject detection results based on mode
            if self.last_valid_box is None:
                return None, orig_img, im_name, None, None, None, None
            
            if self.mode == 'hijack':
                # Apply offset to the x coordinates
                self.current_offset += self.x_offset
                injected_box = self.last_valid_box.clone()
                injected_box[0] += self.current_offset  # x1
                injected_box[2] += self.current_offset  # x2
                # Keep y always the same
                injected_box[1] = self.last_valid_box[1]
                injected_box[3] = self.last_valid_box[3]
            elif self.mode == 'shrink':
                # Calculate center point
                center_x = (self.original_size[0] + self.original_size[2]) / 2
                center_y = (self.original_size[1] + self.original_size[3]) / 2
                
                # Calculate current size reduction
                current_scale = 1.0 - (self.shrink_rate * (frame_idx - start_frame))
                current_scale = max(0.1, current_scale)  # Prevent box from becoming too small
                
                # Calculate new width and height
                original_width = self.original_size[2] - self.original_size[0]
                original_height = self.original_size[3] - self.original_size[1]
                new_width = original_width * current_scale
                new_height = original_height * current_scale
                
                # Create new box centered at the same point
                injected_box = torch.tensor([
                    center_x - new_width/2,  # x1
                    center_y - new_height/2, # y1
                    center_x + new_width/2,  # x2
                    center_y + new_height/2  # y2
                ])
            
            # Ensure box stays within image boundaries
            injected_box[0] = max(0, min(injected_box[0], im_dim[0, 0] - 10))
            injected_box[1] = max(0, min(injected_box[1], im_dim[0, 1] - 10))
            injected_box[2] = max(injected_box[0] + 10, min(injected_box[2], im_dim[0, 0]))
            injected_box[3] = max(injected_box[1] + 10, min(injected_box[3], im_dim[0, 1]))
            
            boxes = injected_box.unsqueeze(0)
            scores = torch.tensor([[0.9]])  # High confidence for injected box
            ids = torch.zeros(scores.shape)

        # Process the box for pose estimation
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)

        return inps, orig_img, im_name, boxes, scores, ids, cropped_boxes


class AdaptiveTester:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        
        # Load pose model
        print(f'Loading pose model from {args.pose_checkpoint}...')
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(args.pose_checkpoint, map_location=args.device))
        self.pose_model.to(args.device)
        self.pose_model.eval()
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        
        # Initialize detector and injector
        self.detector = get_detector(self.args)
        self.det_injector = DetectionInjector(self.detector, self.cfg, self.args)
        
        # Load LSTM model
        print(f'Loading LSTM model from {args.lstm_checkpoint}...')
        checkpoint = torch.load(args.lstm_checkpoint, map_location=args.device)
        self.lstm_model = OneClassLSTM(
            input_size=408,
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.to(args.device)
        self.lstm_model.eval()
        
        # Set up sequence buffer for LSTM input
        self.window_size = args.window_size
        self.sequence_buffer = []
        
        # Initialize helpers for pose processing
        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        
        # Set up visualization thresholds
        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num
        
        # Decision threshold for anomaly detection
        self.rho = 0.5
        
    def process_frame(self, frame_idx, im_name, image):
        start_time = getTime()
        anomaly_score = None
        
        # Process detection and get pose input
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_injector.process(
                im_name, image, frame_idx, self.args.start_frame
            )
            
            # If no detections, return original image
            if inps is None or boxes is None:
                return orig_img, None
            
            # Pose Estimation
            inps = inps.to(self.args.device)
            if self.args.flip:
                inps = torch.cat((inps, torch.flip(inps, [3])))
            
            # Forward pass through the pose model
            hm = self.pose_model(inps)
            
            if self.args.flip:
                from alphapose.utils.transforms import flip_heatmap
                hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
            
            hm = hm.cpu()
            
            # Process heatmap to get coordinates
            hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
            norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
            
            # Get coordinates from heatmap
            pose_coords = []
            pose_scores = []
            
            for i in range(hm.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                        hm[i][self.eval_joints[:-110]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                        hm[i][self.eval_joints[-110:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(hm[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            
            # Apply NMS if needed
            if len(scores) > 1:
                boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
                    boxes, scores, ids, preds_img, preds_scores, self.args.min_box_area
                )
            
            # Format keypoints for LSTM input
            keypoints = []
            for k in range(len(scores)):
                # Extract keypoints and normalize them
                kp = preds_img[k].numpy()
                kp_score = preds_scores[k].numpy()
                height, width = orig_img.shape[:2]
                
                # Normalize coordinates
                normalized_keypoints = []
                for i in range(kp.shape[0]):
                    x = kp[i, 0] / width
                    y = kp[i, 1] / height
                    s = kp_score[i, 0]
                    normalized_keypoints.extend([x, y, s])
                
                keypoints.append(normalized_keypoints)
            
            # Add to sequence buffer
            if keypoints:
                self.sequence_buffer.append(torch.tensor(keypoints[0], dtype=torch.float32))
                
                # Keep only the most recent window_size frames
                if len(self.sequence_buffer) > self.window_size:
                    self.sequence_buffer.pop(0)
                
                # If we have enough frames, run LSTM inference
                if len(self.sequence_buffer) == self.window_size:
                    seq_tensor = torch.stack(self.sequence_buffer).to(self.args.device)
                    seq_tensor = seq_tensor.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        output = torch.sigmoid(self.lstm_model(seq_tensor))
                        anomaly_score = output.item()
            
            # Visualize results
            from alphapose.utils.vis import vis_frame
            # Create a simple opt object to pass to vis_frame
            class SimpleOpt:
                def __init__(self):
                    self.tracking = False
                    self.showbox = True

            opt = SimpleOpt()
            img = vis_frame(orig_img, {
                'imgname': im_name,
                'result': [{
                    'keypoints': preds_img[k],
                    'kp_score': preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k],
                    'idx': ids[k],
                    'bbox': [boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0], boxes[k][3]-boxes[k][1]]
                } for k in range(len(scores))]
            }, opt, self.vis_thres)
            
            # Add anomaly score to the image if available
            if anomaly_score is not None:
                status = "Normal" if anomaly_score >= self.rho else "Anomalous"
                score_text = f"Score: {anomaly_score:.4f} ({status})"
                cv2.putText(img, score_text, (img.shape[1] - 350, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return img, anomaly_score

def test_video(args, cfg):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # Open video file
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up output video
    base_filename = f"adaptive_test_{os.path.basename(args.video)}"
    # Add .mp4 extension if it doesn't have one
    if not base_filename.endswith('.mp4'):
        base_filename = os.path.splitext(base_filename)[0] + '.mp4'
    output_path = os.path.join(args.outdir, base_filename)
    
    # Use mp4v codec for mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tester
    tester = AdaptiveTester(args, cfg)
    
    frame_idx = 0
    results = []
    
    print(f"Processing video with {total_frames} frames...")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Process the frame
        im_name = f"frame_{frame_idx:06d}"
        img_result, anomaly_score = tester.process_frame(frame_idx, im_name, frame)
        
        # Write result to output video
        if img_result is not None:
            out.write(img_result)
        
        # Save result
        if anomaly_score is not None:
            results.append({
                'frame': frame_idx,
                'score': anomaly_score,
                'status': "Normal" if anomaly_score >= tester.rho else "Anomalous"
            })
        
        # Print progress
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    # Release resources
    video.release()
    out.release()
    
    # Save results to JSON
    results_path = os.path.join(args.outdir, f"results_{os.path.basename(args.video)}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {frame_idx} frames")
    print(f"Output saved to {output_path}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VOGUES Adaptive Testing')
    parser.add_argument('--cfg', type=str, required=True, help='AlphaPose config file')
    parser.add_argument('--pose_checkpoint', type=str, required=True, help='AlphaPose checkpoint file')
    parser.add_argument('--lstm_checkpoint', type=str, required=True, help='LSTM model checkpoint file')
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory')
    parser.add_argument('--start_frame', type=int, default=30, help='Frame to start injection')
    parser.add_argument('--x_offset', type=float, default=5.0, help='X-axis offset per step')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for LSTM input')
    parser.add_argument('--detector', type=str, default='yolo', help='Detector name')
    parser.add_argument('--min_box_area', type=int, default=0, help='Min box area to filter out')
    parser.add_argument('--flip', default=False, action='store_true', help='Enable flip testing')
    parser.add_argument('--gpus', type=str, default="0", help='GPUs to use')
    parser.add_argument('--mode', type=str, default='hijack', choices=['hijack', 'shrink'], 
                       help='Injection mode: hijack (move box) or shrink (reduce size)')
    parser.add_argument('--shrink_rate', type=float, default=0.01, 
                       help='Rate at which to shrink the box (only used in shrink mode)')
    
    args = parser.parse_args()
    
    # Set up device
    args.gpus = [int(args.gpus)] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    
    # Update config
    cfg = update_config(args.cfg)
    
    # Run test
    test_video(args, cfg)

