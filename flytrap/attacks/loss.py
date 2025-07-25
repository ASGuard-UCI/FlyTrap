import torch
import torch.nn.functional as F

from ..builder import LOSS


@LOSS.register_module()
class BaseLoss:
    """Base loss class."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, inputs: dict):
        raise NotImplementedError


@LOSS.register_module()
class TVLoss(BaseLoss):
    """Total variation loss."""

    def __init__(self, weight: float = 5e-6):
        super(TVLoss, self).__init__(weight)

    def __call__(self, inputs):
        img = inputs['patch']
        # Compute the difference of horizontally adjacent pixels
        horizontal_tv = torch.pow(img[:, :, :-1] - img[:, :, 1:], 2).mean()

        # Compute the difference of vertically adjacent pixels
        vertical_tv = torch.pow(img[:, :-1, :] - img[:, 1:, :], 2).mean()

        # Combine the two components
        tv_loss = (horizontal_tv + vertical_tv)
        return self.weight * tv_loss


@LOSS.register_module()
class LocalizationLoss(BaseLoss):
    """Localization loss."""

    def __init__(self, weight=1e-2, smooth_label=False):
        super(LocalizationLoss, self).__init__(weight)
        self.smooth_label = smooth_label

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        # shrink the bbox
        if not self.smooth_label:
            gt_bbox[..., 2:].zero_()
        else:
            gt_bbox[..., 2:] = 0.2 * gt_bbox[..., 2:]
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class TargetLocalizationLoss(BaseLoss):
    """Localization loss."""

    def __init__(self, weight=1e-2):
        super(TargetLocalizationLoss, self).__init__(weight)

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        target_bbox = inputs['target_bbox']
        loss = F.l1_loss(pred_bbox, target_bbox)
        return self.weight * loss


@LOSS.register_module()
class TargetLocalizationLossDet(BaseLoss):
    """Localization loss for object detector."""

    def __init__(self, weight=1e-2):
        super(TargetLocalizationLossDet, self).__init__(weight)

    def __call__(self, inputs: dict):
        pred_bbox_det = inputs['pred_bbox_det'] # [N, 8] [image_id, x1, y1, x2, y2, score, label, track_id]
        target_bbox_det = inputs['target_bbox_det'] # [B, 4] [cx, cy, w, h]
        
        batch_idx = pred_bbox_det[:, 0]
        pred_bbox_det = pred_bbox_det[:, 1:5]
        
        # organize target_bbox_det to match the shape of pred_bbox_det
        target_bbox_det_new = torch.zeros_like(pred_bbox_det)
        for i in range(pred_bbox_det.size(0)):
            target_bbox_det_new[i] = target_bbox_det[batch_idx[i].long()]
            
        # convert pred_bbox_det from [x1, y1, x2, y2] to [cx, cy, w, h]
        pred_bbox_det = torch.stack([(pred_bbox_det[:, 0] + pred_bbox_det[:, 2]) / 2,
                                     (pred_bbox_det[:, 1] + pred_bbox_det[:, 3]) / 2,
                                     pred_bbox_det[:, 2] - pred_bbox_det[:, 0],
                                     pred_bbox_det[:, 3] - pred_bbox_det[:, 1]], dim=1)
        
        # only compute loss if center distance is less than 500
        mask = (target_bbox_det_new - pred_bbox_det)[:, :2].norm(dim=-1, keepdim=True) < 500
        
        loss = F.l1_loss(pred_bbox_det * mask, target_bbox_det_new * mask)
        return self.weight * loss


@LOSS.register_module()
class TargetLocalizationLossDetV2(BaseLoss):
    """Localization loss for object detector."""

    def __init__(self, weight=1e-2, top_k=20):
        super(TargetLocalizationLossDetV2, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_bbox_det = inputs['pred_bbox_det'] # [N, 8] [image_id, x1, y1, x2, y2, score, label, track_id]
        target_bbox_det = inputs['target_bbox_det'] # [B, 4] [cx, cy, w, h]
        
        batch_idx = pred_bbox_det[:, 0]
        pred_bbox_det = pred_bbox_det[:, 1:5]
        
        # organize target_bbox_det to match the shape of pred_bbox_det
        target_bbox_det_new = torch.zeros_like(pred_bbox_det)
        for i in range(pred_bbox_det.size(0)):
            target_bbox_det_new[i] = target_bbox_det[batch_idx[i].long()]
            
        # convert pred_bbox_det from [x1, y1, x2, y2] to [cx, cy, w, h]
        pred_bbox_det = torch.stack([(pred_bbox_det[:, 0] + pred_bbox_det[:, 2]) / 2,
                                     (pred_bbox_det[:, 1] + pred_bbox_det[:, 3]) / 2,
                                     pred_bbox_det[:, 2] - pred_bbox_det[:, 0],
                                     pred_bbox_det[:, 3] - pred_bbox_det[:, 1]], dim=1)
        
        # only compute loss if iou > 0.8
        iou = compute_iou(pred_bbox_det.unsqueeze(1), target_bbox_det_new)
        top_k_iou, top_k_idx = torch.topk(iou, self.top_k, dim=0)
        
        loss = F.l1_loss(pred_bbox_det[top_k_idx], target_bbox_det_new[top_k_idx])
        return self.weight * loss

    
@LOSS.register_module()
class SmoothDynamicClassificationLossDet(BaseLoss):
    def __init__(self, weight=0.1, iou_threshold=0.6):
        super(SmoothDynamicClassificationLossDet, self).__init__(weight)
        self.iou_threshold = iou_threshold

    def __call__(self, inputs: dict):
        pred_bbox_det = inputs['pred_bbox_det'] # [N, 8] [image_id, x1, y1, x2, y2, score, label, track_id]
        target_bbox_det = inputs['target_bbox_det'] # [B, 4] [cx, cy, w, h]
        
        batch_idx = pred_bbox_det[:, 0]
        pred_score = pred_bbox_det[:, 5]
        pred_bbox_det = pred_bbox_det[:, 1:5]
        
        target_score = torch.ones_like(pred_score, device=pred_score.device) * 0.95
        
        # organize target_bbox_det to match the shape of pred_bbox_det
        target_bbox_det_new = torch.zeros_like(pred_bbox_det)
        for i in range(pred_bbox_det.size(0)):
            target_bbox_det_new[i] = target_bbox_det[batch_idx[i].long()]
            
        # convert pred_bbox_det from [x1, y1, x2, y2] to [cx, cy, w, h]
        pred_bbox_det = torch.stack([(pred_bbox_det[:, 0] + pred_bbox_det[:, 2]) / 2,
                                     (pred_bbox_det[:, 1] + pred_bbox_det[:, 3]) / 2,
                                     pred_bbox_det[:, 2] - pred_bbox_det[:, 0],
                                     pred_bbox_det[:, 3] - pred_bbox_det[:, 1]], dim=1)
        
        # only compute loss if iou > 0.8
        mask = compute_iou(pred_bbox_det.unsqueeze(1), target_bbox_det_new) > self.iou_threshold
        
        mask = mask.squeeze()
        
        loss = F.binary_cross_entropy(pred_score * mask, target_score * mask)
        return self.weight * loss


@LOSS.register_module()
class SmoothDynamicClassificationLossDetV2(BaseLoss):
    def __init__(self, weight=0.1, top_k=20, target_score=0.95):
        super(SmoothDynamicClassificationLossDetV2, self).__init__(weight)
        self.top_k = top_k
        self.target_score = target_score

    def __call__(self, inputs: dict):
        pred_bbox_det = inputs['pred_bbox_det'] # [N, 8] [image_id, x1, y1, x2, y2, score, label, track_id]
        target_bbox_det = inputs['target_bbox_det'] # [B, 4] [cx, cy, w, h]
        
        batch_idx = pred_bbox_det[:, 0]
        pred_score = pred_bbox_det[:, 5]
        pred_bbox_det = pred_bbox_det[:, 1:5]
        
        target_score = torch.ones_like(pred_score, device=pred_score.device) * self.target_score
        
        # organize target_bbox_det to match the shape of pred_bbox_det
        target_bbox_det_new = torch.zeros_like(pred_bbox_det)
        for i in range(pred_bbox_det.size(0)):
            target_bbox_det_new[i] = target_bbox_det[batch_idx[i].long()]
            
        # convert pred_bbox_det from [x1, y1, x2, y2] to [cx, cy, w, h]
        pred_bbox_det = torch.stack([(pred_bbox_det[:, 0] + pred_bbox_det[:, 2]) / 2,
                                     (pred_bbox_det[:, 1] + pred_bbox_det[:, 3]) / 2,
                                     pred_bbox_det[:, 2] - pred_bbox_det[:, 0],
                                     pred_bbox_det[:, 3] - pred_bbox_det[:, 1]], dim=1)
        
        # only compute loss if iou > 0.8
        iou = compute_iou(pred_bbox_det.unsqueeze(1), target_bbox_det_new)
        top_k_iou, top_k_idx = torch.topk(iou, self.top_k, dim=0)
        
        loss = F.binary_cross_entropy(pred_score[top_k_idx], target_score[top_k_idx])
        return self.weight * loss


@LOSS.register_module()
class LocalizationLossV2(BaseLoss):
    """Localization loss. Set a extreme width and height instead of optimize to 0"""

    def __init__(self, weight=1e-2, shrink_rate=0.3):
        super(LocalizationLossV2, self).__init__(weight)
        self.shrink_rate = shrink_rate

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        # shrink the bbox
        gt_bbox[..., 2:] = gt_bbox[..., 2:] * self.shrink_rate
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class LocalizationLossUnfollow(BaseLoss):
    """Localization loss."""

    def __init__(self, weight=1e-2):
        super(LocalizationLossUnfollow, self).__init__(weight)

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class ClassificationLoss(BaseLoss):
    """Classification loss."""

    def __init__(self, weight=5e-3):
        super(ClassificationLoss, self).__init__(weight)

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        gt_score = torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score, gt_score)
        return self.weight * loss


@LOSS.register_module()
class SmoothClassificationLossUnfollow(BaseLoss):
    """Smoothed classification loss."""

    def __init__(self, weight=5e-3):
        super(SmoothClassificationLossUnfollow, self).__init__(weight)

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        gt_score = 0.05 * torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score, gt_score)
        return self.weight * loss


@LOSS.register_module()
class SmoothClassificationLossUnfollowV2(BaseLoss):
    """Smoothed classification loss. Only compute loss when iou is larger than 0.5."""

    def __init__(self, weight=5e-3, iou_threshold=0.7):
        super(SmoothClassificationLossUnfollowV2, self).__init__(weight)
        self.iou_threshold = iou_threshold

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        gt_bbox = inputs['gt_bbox']
        pred_bbox = inputs['pred_bbox']
        iou = compute_iou(pred_bbox.unsqueeze(1), gt_bbox)
        mask = iou > self.iou_threshold
        gt_score = 0.05 * torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


@LOSS.register_module()
class SmoothClassificationLossUnfollowV3(BaseLoss):
    """Smoothed classification loss. Only compute loss when iou is larger than 0.5."""

    def _loc_mask(self, pred_bbox, gt_bbox):
        """Compute if the shrinking bbox is on the patch"""

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        def to_corners(bbox):
            cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return x1, y1, x2, y2

        pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_bbox)
        gt_x1, gt_y1, gt_x2, gt_y2 = to_corners(gt_bbox)

        # Check if pred_bbox is fully inside gt_bbox
        mask = (gt_x1 <= pred_x1) & (gt_y1 <= pred_y1) & (gt_x2 >= pred_x2) & (gt_y2 >= pred_y2)

        # Convert mask to [B, 1] shape
        mask = mask.unsqueeze(1).float()

        return mask

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        gt_score = 0.05 * torch.ones_like(pred_score, device=pred_score.device)
        mask = self._loc_mask(pred_bbox, gt_bbox)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


@LOSS.register_module()
class DynamicClassificationLoss(BaseLoss):
    """Dynamic classification loss. Only output loss when the location is on the patch"""

    def __init__(self, weight=5e-3):
        super(DynamicClassificationLoss, self).__init__(weight)

    def _loc_mask(self, pred_bbox, gt_bbox):
        """Compute if the shrinking bbox is on the patch"""

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        def to_corners(bbox):
            cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return x1, y1, x2, y2

        pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_bbox)
        gt_x1, gt_y1, gt_x2, gt_y2 = to_corners(gt_bbox)

        # Check if pred_bbox is fully inside gt_bbox
        mask = (gt_x1 <= pred_x1) & (gt_y1 <= pred_y1) & (gt_x2 >= pred_x2) & (gt_y2 >= pred_y2)

        # Convert mask to [B, 1] shape
        mask = mask.unsqueeze(1).float()

        return mask

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        gt_score = torch.ones_like(pred_score, device=pred_score.device)
        mask = self._loc_mask(pred_bbox, gt_bbox)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


@LOSS.register_module()
class SmoothDynamicClassificationLoss(BaseLoss):
    """Dynamic classification loss. Only output loss when the location is on the patch"""

    def __init__(self, weight=5e-3):
        super(SmoothDynamicClassificationLoss, self).__init__(weight)

    def _loc_mask(self, pred_bbox, gt_bbox):
        """Compute if the shrinking bbox is on the patch"""

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        def to_corners(bbox):
            cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return x1, y1, x2, y2

        pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_bbox)
        gt_x1, gt_y1, gt_x2, gt_y2 = to_corners(gt_bbox)

        # Check if pred_bbox is fully inside gt_bbox
        mask = (gt_x1 <= pred_x1) & (gt_y1 <= pred_y1) & (gt_x2 >= pred_x2) & (gt_y2 >= pred_y2)

        # Convert mask to [B, 1] shape
        mask = mask.unsqueeze(1).float()

        return mask

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        gt_score = 0.95 * torch.ones_like(pred_score, device=pred_score.device)
        mask = self._loc_mask(pred_bbox, gt_bbox)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


@LOSS.register_module()
class SmoothDynamicClassificationLossV2(BaseLoss):
    """Dynamic classification loss. Only output loss when the location has high iou to the target shrink bbox
    Please ensure the shrink rate is the same as the one used in LocalizationLossV2"""

    def __init__(self, weight=5e-3, shrink_rate=0.3, iou_threshold=0.7):
        super(SmoothDynamicClassificationLossV2, self).__init__(weight)
        self.shrink_rate = shrink_rate
        self.iou_threshold = iou_threshold

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        gt_bbox[..., 2:] = gt_bbox[..., 2:] * self.shrink_rate
        iou = compute_iou(pred_bbox.unsqueeze(1), gt_bbox)
        mask = iou > self.iou_threshold
        gt_score = 0.9 * torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


@LOSS.register_module()
class SmoothDynamicUnfollowLoss(BaseLoss):
    """Dynamic classification loss. Only output loss when the location is on the patch"""

    def __init__(self, weight=5e-3):
        super(SmoothDynamicUnfollowLoss, self).__init__(weight)

    def _loc_mask(self, pred_bbox, gt_bbox):
        """Compute if the shrinking bbox is on the patch"""

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        def to_corners(bbox):
            cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return x1, y1, x2, y2

        pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_bbox)
        gt_x1, gt_y1, gt_x2, gt_y2 = to_corners(gt_bbox)

        # Check if pred_bbox is fully inside gt_bbox
        mask = (gt_x1 <= pred_x1) & (gt_y1 <= pred_y1) & (gt_x2 >= pred_x2) & (gt_y2 >= pred_y2)

        # Convert mask to [B, 1] shape
        mask = mask.unsqueeze(1).float()

        return mask

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        gt_score = 1e-3 * torch.ones_like(pred_score, device=pred_score.device)
        mask = self._loc_mask(pred_bbox, gt_bbox)
        loss = F.binary_cross_entropy(pred_score * mask, gt_score * mask)
        return self.weight * loss


def compute_iou(pred_bbox, gt_bbox):
    """
    Args:
        pred_bbox: torch.Tensor, [B, N, 4] [cx, cy, w, h]
        gt_bbox: torch.Tensor, [B, 4] [cx, cy, w, h]
    Returns:
        iou: torch.Tensor, [B, N]
    """

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    def to_corners(bbox):
        cx, cy, w, h = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    # Expand gt_bbox to match the shape of pred_bbox
    gt_bbox = gt_bbox.unsqueeze(1).expand(-1, pred_bbox.size(1), -1)

    pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_bbox)
    gt_x1 = gt_bbox[:, :, 0] - gt_bbox[:, :, 2] / 2
    gt_y1 = gt_bbox[:, :, 1] - gt_bbox[:, :, 3] / 2
    gt_x2 = gt_bbox[:, :, 0] + gt_bbox[:, :, 2] / 2
    gt_y2 = gt_bbox[:, :, 1] + gt_bbox[:, :, 3] / 2

    # Calculate intersection coordinates
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate areas of predicted and ground truth boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    # Calculate union area
    union_area = pred_area + gt_area - inter_area

    # Calculate IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou


@LOSS.register_module()
class AlignLocalizationLoss(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2):
        super(AlignLocalizationLoss, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['target_bbox']
        score = inputs['pred_score']
        top_k_idx = torch.topk(score, self.top_k, dim=1)[1]
        pred_bbox = torch.gather(pred_bbox, 1, top_k_idx.unsqueeze(-1).expand(-1, -1, 4))
        # shrink the bbox
        # gt_bbox[..., 2:].zero_()
        gt_bbox = gt_bbox.unsqueeze(1).expand(-1, pred_bbox.size(1), -1)
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class AlignLocalizationLossUnfollow(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2):
        super(AlignLocalizationLossUnfollow, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        score = inputs['pred_score']
        top_k_idx = torch.topk(score, self.top_k, dim=1)[1]
        pred_bbox = torch.gather(pred_bbox, 1, top_k_idx.unsqueeze(-1).expand(-1, -1, 4))
        # shrink the bbox
        gt_bbox = gt_bbox.unsqueeze(1).expand(-1, pred_bbox.size(1), -1)
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class AlignClassificationLoss(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2, smooth_label=True):
        super(AlignClassificationLoss, self).__init__(weight)
        self.top_k = top_k
        self.smooth_label = smooth_label

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        top_k_idx = torch.topk(pred_score, self.top_k, dim=1)[1]
        pred_score = torch.gather(pred_score, 1, top_k_idx)
        # shrink the bbox
        if self.smooth_label:
            gt_score = 0.8 * torch.ones_like(pred_score, device=pred_score.device)
        else:
            gt_score = torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score, gt_score)
        return self.weight * loss


@LOSS.register_module()
class AlignClassificationLossUnfollow(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2):
        super(AlignClassificationLossUnfollow, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        top_k_idx = torch.topk(pred_score, self.top_k, dim=1)[1]
        pred_score = torch.gather(pred_score, 1, top_k_idx)
        # shrink the bbox
        gt_score = 0.05 * torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score, gt_score)
        return self.weight * loss


@LOSS.register_module()
class AlignLocalizationLossIoU(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2):
        super(AlignLocalizationLossIoU, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        iou = compute_iou(pred_bbox, gt_bbox)
        top_k_idx = torch.topk(iou, self.top_k, dim=1)[1]
        pred_bbox = torch.gather(pred_bbox, 1, top_k_idx.unsqueeze(-1).expand(-1, -1, 4))
        # shrink the bbox
        gt_bbox[..., 2:].zero_()
        gt_bbox = gt_bbox.unsqueeze(1).expand(-1, pred_bbox.size(1), -1)
        loss = F.l1_loss(pred_bbox, gt_bbox)
        return self.weight * loss


@LOSS.register_module()
class AlignClassificationLossIoU(BaseLoss):
    """Used for SiamRPN, predict multiple bbox based on anchors"""

    def __init__(self, top_k=50, weight=1e-2):
        super(AlignClassificationLossIoU, self).__init__(weight)
        self.top_k = top_k

    def __call__(self, inputs: dict):
        pred_score = inputs['pred_score']
        pred_bbox = inputs['pred_bbox']
        gt_bbox = inputs['gt_bbox']
        iou = compute_iou(pred_bbox, gt_bbox)
        top_k_idx = torch.topk(iou, self.top_k, dim=1)[1]
        pred_score = torch.gather(pred_score, 1, top_k_idx)
        # shrink the bbox
        gt_score = 0.9 * torch.ones_like(pred_score, device=pred_score.device)
        loss = F.binary_cross_entropy(pred_score, gt_score)
        return self.weight * loss


@LOSS.register_module()
class NaturalConstraintLoss(BaseLoss):
    """Natural constraint loss. Make the patch looks like natural image."""

    def __init__(self, weight=5e-6):
        super(NaturalConstraintLoss, self).__init__(weight)

    def __call__(self, inputs: dict):
        patch = inputs['patch']
        natural_img = inputs['natural_img']
        loss = F.mse_loss(patch, natural_img)
        return self.weight * loss
    
    
@LOSS.register_module()
class PoseRegressionLoss(BaseLoss):
    def __init__(self, weight = 1):
        super().__init__(weight)
        
    def __call__(self, inputs):
        pred_pose = inputs['pred_pose'] # [B, C, H, W]
        target_pose = inputs['target_pose'] # [C, H, W]
        target_pose = target_pose.unsqueeze(0).repeat(pred_pose.size(0), 1, 1, 1)
        
        pred_pose = F.sigmoid(pred_pose)
        target_pose = F.sigmoid(target_pose)
        
        loss = F.binary_cross_entropy(pred_pose, target_pose)
        return self.weight * loss
        
        
@LOSS.register_module()
class PoseMSELoss(BaseLoss):
    def __init__(self, weight = 1):
        super().__init__(weight)
        
    def __call__(self, inputs):
        pred_pose = inputs['pred_pose'] # [B, C, H, W]
        target_pose = inputs['target_pose'] # [C, H, W]
        target_pose = target_pose.unsqueeze(0).repeat(pred_pose.size(0), 1, 1, 1)
        
        loss = F.mse_loss(pred_pose, target_pose)
        return self.weight * loss
        