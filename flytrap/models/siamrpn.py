import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pysot.pysot.models.backbone import get_backbone
from models.pysot.pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from models.pysot.pysot.models.neck import get_neck
from models.pysot.pysot.tracker.base_tracker import SiameseTracker
from models.pysot.pysot.utils.anchor import Anchors
from ..builder import MODEL


@MODEL.register_module()
class ModelBuilder(nn.Module):
    """Customized model builder for SiamMask
    Originated from: pysot/models/model_builder.py"""

    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()

        self.cfg = cfg
        # build anchor
        self.score_size = (self.cfg.TRACK.INSTANCE_SIZE - self.cfg.TRACK.EXEMPLAR_SIZE) // \
                          self.cfg.ANCHOR.STRIDE + 1 + self.cfg.TRACK.BASE_SIZE
        self.anchors = torch.tensor(self.generate_anchor(self.score_size))

        # build backbone
        self.backbone = get_backbone(self.cfg.BACKBONE.TYPE,
                                     **self.cfg.BACKBONE.KWARGS)

        # build adjust layer
        if self.cfg.ADJUST.ADJUST:
            self.neck = get_neck(self.cfg.ADJUST.TYPE,
                                 **self.cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(self.cfg.RPN.TYPE,
                                     **self.cfg.RPN.KWARGS)

        # build mask head
        if self.cfg.MASK.MASK:
            self.mask_head = get_mask_head(self.cfg.MASK.TYPE,
                                           **self.cfg.MASK.KWARGS)

            if self.cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(self.cfg.REFINE.TYPE)

        msg = self.load_state_dict(torch.load(self.cfg.CKPT))
        print(msg)

    def _convert_bbox(self, delta, anchor):
        """
        Args:
            delta: torch.Tensor, [B, 4*A, H, W]
            anchor: np.ndarray, [A*H*W, 4] [cx, cy, w, h]
        """
        B = delta.shape[0]
        # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.view(B, 4, -1)
        anchor = anchor.to(delta.device)

        delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        delta[:, 2, :] = torch.exp(delta[:, 2, :]) * anchor[:, 2]
        delta[:, 3, :] = torch.exp(delta[:, 3, :]) * anchor[:, 3]
        # [B, A, 4*H*W]
        return delta

    def _convert_score(self, score):
        # [B, 2*A, H, W]
        B = score.shape[0]
        score = score.view(B, 2, -1).permute(0, 2, 1)
        score = F.softmax(score, dim=2)[..., 1]
        return score

    def generate_anchor(self, score_size):
        anchors = Anchors(self.cfg.ANCHOR.STRIDE,
                          self.cfg.ANCHOR.RATIOS,
                          self.cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if self.cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if self.cfg.MASK.MASK else None
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def _xyxy2cxcywh(self, bbox):
        """
        Args:
            bbox: torch.Tensor, [B, N, 4]
        """
        cxcy = (bbox[:, :, :2] + bbox[:, :, 2:]) / 2
        wh = bbox[:, :, 2:] - bbox[:, :, :2]
        return torch.cat([cxcy, wh], dim=2)

    def forward(self, template, search, **kwargs):
        """ only used in training
        """

        B, _, H, W = search.size()
        cx = H / 2
        cy = W / 2
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        # cls: [B, 2*A, H, W]
        # loc: [B, 4*A, H, W]
        cls, loc = self.rpn_head(zf, xf)

        loc = self._convert_bbox(loc, self.anchors)  # [B, 4, A*H*W] [cx, cy, w, h]
        # original point is [cx, cy]
        # shift coordinate to start from 0
        loc = loc + torch.tensor([cx, cy, 0, 0], device=loc.device).view(-1, 4, 1)
        # convert to cxcywh in relative coordinate
        # loc = self._xyxy2cxcywh(loc.permute(0, 2, 1))  # [B, A*H*W, 4]
        loc = loc.permute(0, 2, 1)
        loc = loc / torch.tensor([W, H, W, H], device=loc.device)

        cls = self._convert_score(cls)

        outputs = [
            dict(
                pred_boxes=loc, # [cx, cy, w, h]
                pred_scores=cls,
            ),
        ]

        if self.cfg.MASK.MASK:
            raise NotImplementedError
        return outputs


@MODEL.register_module()
class SiamRPNTracker(SiameseTracker):
    def __init__(self, cfg):
        super(SiamRPNTracker, self).__init__()
        self.model = MODEL.build(cfg)
        self.cfg = cfg.cfg
        self.score_size = (self.cfg.TRACK.INSTANCE_SIZE - self.cfg.TRACK.EXEMPLAR_SIZE) // \
                          self.cfg.ANCHOR.STRIDE + 1 + self.cfg.TRACK.BASE_SIZE
        self.anchor_num = len(self.cfg.ANCHOR.RATIOS) * len(self.cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        # since pysot use cuda by default
        # hard-code here instead of choose device
        self.model.cuda()
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(self.cfg.ANCHOR.STRIDE,
                          self.cfg.ANCHOR.RATIOS,
                          self.cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def initialize(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        bbox = bbox['init_bbox']
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'target_bbox': list(map(float, bbox)),  # [x, y, w, h]
            'target_score': float(best_score)
        }
