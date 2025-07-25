# This script contains Alarm defense
# e.g. VOGUES or other spatial-temporal consistency based defense

# This can be argumented back:
# 1. In the unwill tracking scenario, the pilot trys to follow the target person as much as possible
# Therefore, it's not applicable if the pilot adopt a strong defense mechanism, which hamper the benign performance
# e.g., if the IOU threshold is too high, the ATT drone is easily losing track

import numpy as np
import torch

from ..builder import ALARM, MODEL

@ALARM.register_module()
class VOGUES():
    def __init__(self, detector, pose_estimator, iou_threshold=0.5):
        self.detector = MODEL.build(detector)
        self.pose_estimator = MODEL.build(pose_estimator)
        self.iou_threshold = iou_threshold

    def _iou_compute(self, tracker_bbox, det_bbox):
        """
        Args:
            tracker_bbox: np.ndarray, shape (4,), dtype float
            det_bbox: np.ndarray, shape (N, 4), dtype float
        Returns:
            iou: np.ndarray, shape (N,), dtype float
        """
        x1 = np.maximum(tracker_bbox[0], det_bbox[:, 0])
        y1 = np.maximum(tracker_bbox[1], det_bbox[:, 1])
        x2 = np.minimum(tracker_bbox[0] + tracker_bbox[2], det_bbox[:, 2])
        y2 = np.minimum(tracker_bbox[1] + tracker_bbox[3], det_bbox[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        tracker_area = tracker_bbox[2] * tracker_bbox[3]
        det_area = (det_bbox[:, 2] - det_bbox[:, 0]) * (det_bbox[:, 3] - det_bbox[:, 1])

        iou = inter_area / (tracker_area + det_area - inter_area)
        return iou
        
        
    def __call__(self, tracker_output, img):
        """
        Args:
            img: np.ndarray, shape (H, W, C), dtype uint8
        """
        tracker_bbox = np.array(tracker_output['target_bbox']) # [x1, y1, w, h]
        tracker_score = tracker_output['target_score']
        
        H, W = img.shape[:2]
        
        orig_shape = torch.tensor(np.array([W, H, W, H]))[None, ]
        orig_shape = torch.tensor(orig_shape)
        input_img = self.detector.image_preprocess(img)
        detector_output = self.detector.images_detection(input_img, orig_shape)
        bbox = detector_output[:, 1:5].cpu().numpy()
        ious = self._iou_compute(tracker_bbox, bbox)
        return float(ious.max())


@ALARM.register_module()
class KalmanFilter():
    def __init__(self, iou_threshold=0.5):
        pass

    def __call__(self, tracker_output, img):
        pass
    
