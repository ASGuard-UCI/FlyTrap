import os
import cv2
import torch
import warnings
import numpy as np
import uuid
from collections import deque
from argparse import ArgumentParser, Namespace
from scipy.optimize import linear_sum_assignment


from alphapose.models import builder
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.config import update_config
from detector.apis import get_detector
from trackers.tracker_cfg import cfg as tcfg
from vogues.model import OneClassLSTM

# Import SingleImageAlphaPose from demo_api
from scripts.demo_api import SingleImageAlphaPose


class VOGUES(object):
    """Reproduction of USENIX Security 2024: VOGUES: Validation of Object Guise using Estimated Components
    Args:
        cfg: config file for pose model
        checkpoint_path: path to the pose model checkpoint
        use_lstm: whether to use LSTM for temporal anomaly detection
        lstm_checkpoint_path: path to the LSTM checkpoint
        device: device to run the model
        multi_obj: whether to handle multiple objects
        iou_threshold: threshold for IoU matching
        detector_name: name of the detector to use
    """
    def __init__(self, 
                 cfg: str, 
                 checkpoint_path: str, 
                 use_lstm: bool = False,
                 lstm_checkpoint_path: str = None,
                 device: str = "cuda",
                 multi_obj: bool = True,
                 iou_threshold: float = 0.5,
                 history_length: int = 10,
                 detector_name: str = "yolo",
                 save_img: bool = False,
                 vis: bool = False,
                 showbox: bool = False):
        
        cfg = update_config(cfg)
        
        # Initialize AlphaPose API
        args = Namespace()
        args.cfg = cfg
        args.checkpoint = checkpoint_path
        args.detector = detector_name
        args.device = torch.device(device)
        args.gpus = [0] if device == "cuda" else [-1]
        args.inputimg = ""  # Will be set in validate method
        args.save_img = save_img
        args.vis = vis
        args.showbox = showbox
        args.profile = False
        args.format = "coco"
        args.min_box_area = 0
        args.eval = False
        args.flip = False
        args.debug = False
        args.vis_fast = False
        args.pose_flow = False
        args.pose_track = False
        args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'
        
        # Initialize the AlphaPose API
        self.alphaPose = SingleImageAlphaPose(args, cfg)
        
        # LSTM for temporal anomaly detection
        if use_lstm:
            # TODO: hardcoded the dimension
            self.lstm_model = OneClassLSTM(input_size=408, hidden_size=128, num_layers=1)
            print('Loading LSTM model from %s...' % (lstm_checkpoint_path,))
            lstm_checkpoint = torch.load(lstm_checkpoint_path, map_location=device)
            self.lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
            self.lstm_model.to(device)
            self.lstm_model.eval()
        else:
            self.lstm_model = None
            
        self.device = device
        self.multi_obj = multi_obj
        self.iou_threshold = iou_threshold
        
        # Initialize pose history as a dictionary of deques
        self.pose_history = {}
        self.history_length = history_length
        
        # Store the cfg
        self.cfg = cfg
        
        self.current_pose = None
        self.current_img = None
        self.max_iou = 0.0
        self.det_bbox = None
        self.lstm_score = 0.0
        
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        Args:
            box1, box2: [x1, y1, x2, y2]
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def vis(self):
        """
        Visualize the bounding box on the image
        """
        img = self.current_img.copy()
        pose = self.current_pose
        return self.alphaPose.vis(img, pose)

    def info(self):
        """
        Get the information of the current pose
        """
        output_dict = {
            'max_iou': float(self.max_iou),
            'det_bbox': self.det_bbox.tolist(),
            'lstm_score': float(self.lstm_score)
        }
        return output_dict

    def validate(self, img, track_results=None, track_id=None) -> bool:
        """
        Validate the detection results
        Args:
            img (np.ndarray): [H, W, 3] in BGR format
            track_results (list): [x1, y1, w, h], track results from the tracker, if None, skip spatial check
            track_id (int): track id to check, required if multi_obj is False
        Returns:
            bool: True if anomaly detected (alarm), False otherwise
        """
        # Step 1: Run object detection and pose estimation using AlphaPose API
        if img is None:
            warnings.warn("No image provided")
            return False
            
        H, W, _ = img.shape
        
        try:
            self.current_img = img.copy()
            # Create a temporary image file to use with the API
            tmp_img_path = f"_temp_image_{uuid.uuid4()}.jpg"
            cv2.imwrite(tmp_img_path, img)
            
            # Process the image using AlphaPose API
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_results = self.alphaPose.process(tmp_img_path, rgb_img)
            # Used for visualization
            self.current_pose = pose_results
            # Remove the temporary image
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            
            # Check if detection or pose estimation failed
            if pose_results is None or len(pose_results['result']) == 0:
                warnings.warn("No detection results found")
                return False
            
            # Convert AlphaPose results to our format
            boxes = []
            keypoints = []
            kp_scores = []
            proposal_scores = []
            
            for person in pose_results['result']:
                # The bbox format in AlphaPose is [x1, y1, width, height]
                # Convert to [x1, y1, x2, y2]
                x1, y1, width, height = person['bbox']
                boxes.append([x1, y1, x1 + width, y1 + height])
                
                # Get keypoints and scores
                # Normalize keypoints to [0, 1] for LSTM input
                keypoints.append(person['keypoints'].numpy() / np.array([W, H]))
                kp_scores.append(person['kp_score'].numpy())
                proposal_scores.append(person['proposal_score'].numpy())
            # Format results in the same structure as before
            pose_det_results = {
                'boxes': torch.tensor(boxes).float(),
                'ids': torch.arange(len(boxes)),   # TODO: add ids
                'keypoints': torch.tensor(keypoints).float(),
                'kp_scores': torch.tensor(kp_scores).float(),
                'proposal_scores': torch.tensor(proposal_scores).float()
            }
            
        except Exception as e:
            warnings.warn(f"Error during detection or pose estimation: {str(e)}")
            return False
        
        # Spatial consistency check
        if track_results is not None and pose_det_results['boxes'] is not None:
            # Convert track_results to [x1, y1, x2, y2]
            track_boxes = np.array([track_results[0], track_results[1], 
                           track_results[0] + track_results[2], track_results[1] + track_results[3]])
            track_boxes = track_boxes.reshape(-1, 4)
            pose_boxes = pose_det_results['boxes'].cpu().numpy()

                
            # Case 1: multi_obj is True
            if self.multi_obj:
                assert False, "multi_obj is not supported yet"
                assert track_id is not None, "track_id must be provided when multi_obj is True"
                
                # If detection counts don't match, raise alarm
                if len(track_boxes) != len(pose_boxes):
                    return True
                
                # Bipartite matching
                cost_matrix = np.zeros((len(track_boxes), len(pose_boxes)))
                for i, track_box in enumerate(track_boxes):
                    for j, pose_box in enumerate(pose_boxes):
                        # Calculate IoU and convert it to a cost (1 - IoU)
                        cost_matrix[i, j] = 1.0 - self.calculate_iou(track_box, pose_box)
                
                # Use Hungarian algorithm for optimal matching
                track_indices, pose_indices = linear_sum_assignment(cost_matrix)
                
                # Check IoU of matched pairs
                for track_idx, pose_idx in zip(track_indices, pose_indices):
                    if 1.0 - cost_matrix[track_idx, pose_idx] < self.iou_threshold:
                        return True  # Alarm if any IoU is below threshold
            
            # Case 2: multi_obj is False
            else:
                assert len(track_boxes) == 1, "track_results should contain only one object when multi_obj is False"
                
                track_box = track_boxes[0]
                max_iou = 0.0
                idx = 0
                
                # Find the max IoU
                for i, pose_box in enumerate(pose_boxes):
                    iou = self.calculate_iou(track_box, pose_box)
                    if iou > max_iou:
                        max_iou = iou
                        idx = i
                
                self.max_iou = max_iou
                self.det_bbox = pose_boxes[idx: idx+1]
                
                # If max IoU is below threshold, raise alarm
                if max_iou < self.iou_threshold:
                    return True
        
        # Temporal consistency check with LSTM (if available)
        if self.lstm_model is not None and pose_det_results['keypoints'] is not None:
            pose_keypoints = pose_det_results['keypoints'][idx: idx+1].cpu().numpy()
            pose_scores = pose_det_results['kp_scores'][idx: idx+1].cpu().numpy()
            pose_input = np.concatenate([pose_keypoints, pose_scores], axis=-1)
            bs = pose_input.shape[0]
            pose_input = pose_input.reshape(bs, -1)
            
            # Get track IDs based on multi_obj setting
            if self.multi_obj:
                raise NotImplementedError("multi_obj is not supported yet")
            else:
                track_ids = [track_id] if track_id is not None else [0]
            
            # Process each track
            for idx, tid in enumerate(track_ids):
                # Initialize history queue if not exists
                if tid not in self.pose_history:
                    self.pose_history[tid] = deque(maxlen=self.history_length)
                
                # Get keypoints for current track
                flat_keypoints = pose_input[idx] if idx < len(pose_input) else None
                
                # Skip if no keypoints
                if flat_keypoints is None:
                    continue
                
                # Add to history
                self.pose_history[tid].append(flat_keypoints)
                
                # Check if we have enough history
                if len(self.pose_history[tid]) == self.history_length:
                    # Convert history to tensor
                    sequence = np.array(list(self.pose_history[tid]))
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get LSTM prediction
                    with torch.no_grad():
                        output = torch.sigmoid(self.lstm_model(sequence_tensor))
                    
                    assert len(output) == 1, "LSTM output should be a single value for SOT"
                    self.lstm_score = output.item()
                    # Check if score is below threshold
                    if output.item() < 0.5:
                        return True  # Temporal anomaly detected
        
        # No anomalies detected
        return False


if __name__ == "__main__":
    # Configuration for AlphaPose
    cfg = 'configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml'
    checkpoint_path = 'pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth'
    use_lstm = True
    lstm_checkpoint_path = 'model.pt'
    device = 'cuda'
    multi_obj = False
    iou_threshold = 0.5
    detector_name = 'yolo'
    
    # Create VOGUES instance
    vogues = VOGUES(cfg, checkpoint_path, use_lstm, lstm_checkpoint_path, device, multi_obj, iou_threshold, detector_name)
    
    # Process a folder of images
    folder = 'data/dataset_v4.0/eval/person1_bareground1_instance2'
    img_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    img_list.sort()
    
    # Process each image
    for img_path in img_list:
        img = cv2.imread(img_path)
        result = vogues.validate(img)
        print(f"Image: {img_path}, Anomaly detected: {result}")