from typing import List

from ..builder import METRICS


@METRICS.register_module()
class AttackSuccessRate:
    def __init__(self,
                 score_threshold: int = 0.5,
                 area_threshold: int = 0.8):
        self.score_threshold = score_threshold
        self.area_threshold = area_threshold

    def _compute_single(self, data):
        successful_attacks = 0

        for frame in data:
            # Extract target score and bounding boxes
            target_score = frame['out']['target_score']
            out_bbox = frame['out']['target_bbox']
            gt_bbox = frame['gt']['target_bbox']

            # Compute areas of bounding boxes
            out_area = out_bbox[2] * out_bbox[3]
            gt_area = gt_bbox[2] * gt_bbox[3]

            # check center of the bounding box
            x1, y1, w, h = out_bbox
            center_x = x1 + w / 2
            center_y = y1 + h / 2
            x1, y1, w, h = gt_bbox
            x2 = x1 + w
            y2 = y1 + h
            # TODO: [ ] modify the metric: the condition should be the bbox is fully inside the gt_bbox
            if center_x >= x1 and center_x <= x2 and center_y >= y1 and center_y <= y2:
                # Check success conditions
                if target_score > self.score_threshold and out_area < gt_area * self.area_threshold:
                    successful_attacks += 1

        total_frames = len(data)
        success_rate = successful_attacks / total_frames if total_frames > 0 else 0

        return success_rate

    def __call__(self, results):
        metrics = {}
        videos = list(results.keys())
        for video in videos:
            metrics[video] = self._compute_single(results[video])
        metrics['mean'] = sum(metrics.values()) / len(metrics)
        return metrics


@METRICS.register_module()
class AttackSuccessRateUnfollow:
    def __init__(self,
                 score_threshold: int = 0.4):
        self.score_threshold = score_threshold

    def _compute_single(self, data):
        successful_attacks = 0

        for frame in data:
            # Extract target score and bounding boxes
            target_score = frame['out']['target_score']
            if target_score < self.score_threshold:
                successful_attacks += 1

        total_frames = len(data)
        success_rate = successful_attacks / total_frames if total_frames > 0 else 0

        return success_rate

    def __call__(self, results):
        metrics = {}
        videos = list(results.keys())
        for video in videos:
            metrics[video] = self._compute_single(results[video])
        metrics['mean'] = sum(metrics.values()) / len(metrics)
        return metrics


@METRICS.register_module()
class MeanAttackSuccessRate:
    def __init__(self, 
                 score_threshold: List[float],
                 area_threshold: List[float]):
        self.score_threshold = score_threshold
        self.area_threshold = area_threshold
        assert len(score_threshold) == len(area_threshold)

    def _compute_single(self, data, score_threshold, area_threshold):
        successful_attacks = 0

        for frame in data:
            # Extract target score and bounding boxes
            target_score = frame['out']['target_score']
            out_bbox = frame['out']['target_bbox']
            gt_bbox = frame['gt']['target_bbox']

            # Compute areas of bounding boxes
            out_area = out_bbox[2] * out_bbox[3]
            gt_area = gt_bbox[2] * gt_bbox[3]

            # check center of the bounding box
            x1, y1, w, h = out_bbox
            center_x = x1 + w / 2
            center_y = y1 + h / 2
            x1, y1, w, h = gt_bbox
            x2 = x1 + w
            y2 = y1 + h
            if center_x >= x1 and center_x <= x2 and center_y >= y1 and center_y <= y2:
                # Check success conditions
                if target_score > score_threshold and out_area < gt_area * area_threshold:
                    successful_attacks += 1

        total_frames = len(data)
        success_rate = successful_attacks / total_frames if total_frames > 0 else 0
        return success_rate

    def __call__(self, results):
        metrics = {}
        videos = list(results.keys())
        for video in videos:
            metrics[video] = 0
            for score_threshold, area_threshold in zip(self.score_threshold, self.area_threshold):
                metrics[video] += self._compute_single(results[video], score_threshold, area_threshold)
            metrics[video] /= len(self.score_threshold)            
        metrics['mean'] = sum(metrics.values()) / len(metrics)
        return metrics
    
    
@METRICS.register_module()
class MeanAttackSuccessRateUnfollow:
    def __init__(self, 
                 score_threshold: List[float]):
        self.score_threshold = score_threshold

    def _compute_single(self, data, score_threshold):
        successful_attacks = 0

        for frame in data:
            # Extract target score and bounding boxes
            target_score = frame['out']['target_score']
            if target_score < score_threshold:
                successful_attacks += 1

        total_frames = len(data)
        success_rate = successful_attacks / total_frames if total_frames > 0 else 0
        return success_rate

    def __call__(self, results):
        metrics = {}
        videos = list(results.keys())
        for video in videos:
            metrics[video] = 0
            for score_threshold in self.score_threshold:
                metrics[video] += self._compute_single(results[video], score_threshold)
            metrics[video] /= len(self.score_threshold)            
        metrics['mean'] = sum(metrics.values()) / len(metrics)
        return metrics