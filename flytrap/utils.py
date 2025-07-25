import random

import cv2
import numpy as np
import torch
import mmcv

from .attacks.utils import clip_pixel_inplace, clip_circle_inplace


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_patch(patch_size: int, img_config: dict):
    h, w = patch_size, patch_size
    mean = torch.FloatTensor(img_config["mean"])
    std = torch.FloatTensor(img_config["std"])
    patch = torch.randn(3, h, w) * std.view(3, 1, 1) + mean.view(3, 1, 1)
    clip_pixel_inplace(patch)
    clip_circle_inplace(patch)
    return patch


def load_patch(patch_path: str):
    print('Loading patch from: ', patch_path)
    # use BGR
    patch = cv2.imread(patch_path)
    # patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = torch.FloatTensor(patch).permute(2, 0, 1)
    return patch


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x
    b = [x1, y1, x1 + w, y1 + h]
    return np.array(b)


def box_xyxy_to_cxcywh_tensor(x):
    """
    Args:
        x: torch.Tensor, shape=(B, 4) [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = x.unbind(dim=-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)
