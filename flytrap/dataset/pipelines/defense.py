# Transforamtion used for defense

from typing import List

import cv2
import numpy as np
from mmcv.transforms import TRANSFORMS


@TRANSFORMS.register_module()
class JPEGCompression:
    def __init__(self, quality=10) -> None:
        self.quality = quality

    def __call__(self, img: np.ndarray):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, img = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(img, 1)
        return img
    

@TRANSFORMS.register_module()
class BitDepthReduction:
    def __init__(self, bit_depth=4) -> None:
        self.bit_depth = bit_depth

    def __call__(self, img: np.ndarray):
        # img: float32 [0, 255]
        img = img.astype(np.uint8)
        
        # Perform bit depth reduction
        img = img // (2 ** (8 - self.bit_depth))
        img = img * (2 ** (8 - self.bit_depth))
        
        # Ensure values are clipped within the valid range
        img = np.clip(img, 0, 255)
        
        return img
        
    

@TRANSFORMS.register_module()
class GaussianNoiseDefense:
    def __init__(self, mean=0, var=0.1) -> None:
        self.mean = mean
        self.var = var
        
    def __call__(self, img: np.ndarray):
        # img: float32 [0, 255]
        img = img / 255.0
        noise = np.random.normal(self.mean, self.var, img.shape)
        img = img + noise
        img = np.clip(img, 0.0, 1.0)
        img = img * 255.0
        return img
            

@TRANSFORMS.register_module()
class MedianBlur:
    def __init__(self, kernel_size=20) -> None:
        self.kernel_size = int(kernel_size)

    def __call__(self, img: np.ndarray):
        img = img.astype(np.uint8)
        img = cv2.medianBlur(img, self.kernel_size)
        return img