import math
import random

import numpy as np
import torch


def clip_circle_inplace(patch: torch.Tensor):
    """Clip the shape of the patch to a circle.
    Args:
        patch: The patch to be clipped. [C, H, W]
    """
    h, w = patch.size()[1:]
    assert h == w, "Patch should be square"
    radius = h // 2
    center = (h // 2, w // 2)
    for width in range(w):
        for height in range(h):
            if (width - center[0]) ** 2 + (height - center[1]) ** 2 > radius ** 2:
                patch[:, height, width].data.zero_()


def clip_pixel_inplace(x):
    """Clip the pixel values of the image tensor x between min and max for each channel, in-place.

    Args:
        x (torch.Tensor): The image tensor with shape [3, H, W].
        img_config (dict): A dictionary containing the 'mean' and 'std' for normalization.
    """
    # 1.0 ensure the patch is not full black
    # which might influence the applyer
    x.data.clamp_(1.0, 255.0)


class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        if warmup_steps > 0:
            self.initial_lr = optimizer.param_groups[0]['lr'] / warmup_steps
        else:
            self.initial_lr = optimizer.param_groups[0]['lr']
        self.lr = optimizer.param_groups[0]['lr']
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=False)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.initial_lr * (self.last_epoch + 1)
        else:
            cos_decay = 0.5 * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            lr = (self.lr - self.eta_min) * cos_decay + self.eta_min
        return lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
