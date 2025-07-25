from typing import List

import numpy as np
import torch
import torchvision
from mmcv.transforms import TRANSFORMS


@TRANSFORMS.register_module()
class GaussianNoise():
    def __init__(self, img_keys: List[str], mean=0, var=5) -> None:
        self.img_keys = img_keys
        self.mean = mean
        self.var = var

    def _call_single(self, img):
        type_img = img.dtype
        random_noise = np.random.randn(*img.shape) * self.var + self.mean
        img = img + random_noise.astype(type_img)
        img = np.clip(img, 0, 255)
        return img

    def __call__(self, results):
        """Support List or single image"""
        for key in self.img_keys:
            img = results[key]
            if isinstance(img, List):
                results[key] = [self._call_single(i) for i in img]
            else:
                results[key] = self._call_single(img)
        return results


@TRANSFORMS.register_module()
class Brightness:
    def __init__(self, factor=0.1):
        self.factor = factor
        self.transform = torchvision.transforms.ColorJitter(brightness=self.factor)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class Contrast:
    def __init__(self, factor=0.1):
        self.factor = factor
        self.transform = torchvision.transforms.ColorJitter(contrast=self.factor)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class Saturation:
    def __init__(self, factor=0.1):
        self.factor = factor
        self.transform = torchvision.transforms.ColorJitter(saturation=self.factor)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class Hue:
    def __init__(self, factor=0.1):
        self.factor = factor
        self.transform = torchvision.transforms.ColorJitter(hue=self.factor)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class PatchNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        device = img.device
        img = (img - self.mean.to(device)) / self.std.to(device)
        return img


@TRANSFORMS.register_module()
class PatchUnnormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        device = img.device
        img = img * self.std.to(device) + self.mean.to(device)
        return img
