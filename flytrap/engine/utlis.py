import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math

from models.AlphaPose.alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
from models.AlphaPose.alphapose.utils.transforms import get_affine_transform, im_to_torch


def letterbox_images(imgs, bboxes, inp_dim):
    """
    Resize a batch of images with unchanged aspect ratio using padding,
    and adjust the bounding box coordinates accordingly.

    Args:
        imgs (np.ndarray):  A NumPy array of shape (B, H, W, C).
        bboxes (np.ndarray): A NumPy array of shape (B, 4) for bounding boxes,
                             in the format [cx, cy, w, h].
        inp_dim (tuple): (w, h) specifying the desired output size.

    Returns:
        batched_canvas (np.ndarray): A NumPy array of shape (B, h, w, 3)
                                     with each image letterboxed.
        new_bboxes (np.ndarray): A NumPy array of shape (B, 4), with each
                                 bounding box transformed to the new scale
                                 and offset.
    """
    B = imgs.shape[0]
    w, h = inp_dim

    # Prepare an empty canvas for the entire batch (128 for padding)
    # Shape = (batch, h, w, 3)
    batched_canvas = np.full((B, h, w, 3), 128, dtype=imgs.dtype)

    # Array to store the updated bounding boxes
    new_bboxes = np.zeros_like(bboxes, dtype=np.float32)  # shape (B,4)

    for i in range(B):
        img = imgs[i]
        img_h, img_w = img.shape[:2]

        # Compute scaling factors
        scale = min(w / img_w, h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize the current image
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Calculate the padding (offset)
        dw = (w - new_w) // 2
        dh = (h - new_h) // 2

        # Place the resized image onto the canvas
        batched_canvas[i, dh:dh+new_h, dw:dw+new_w, :] = resized_image

        # ---- Update bounding box coordinates ----
        # original bbox: [cx, cy, bw, bh]
        cx, cy, bw, bh = bboxes[i]

        # Scale the bounding box center and size
        cx_scaled = cx * scale
        cy_scaled = cy * scale
        bw_scaled = bw * scale
        bh_scaled = bh * scale

        # Add the offsets to the center coordinates
        cx_new = cx_scaled + dw
        cy_new = cy_scaled + dh

        new_bboxes[i] = [cx_new, cy_new, bw_scaled, bh_scaled]

    return batched_canvas, new_bboxes


def prep_frames(imgs, bboxes, inp_dim):
    """
    Prepare a batch of images and their bounding boxes for input to a network.

    Args:
        imgs (np.ndarray): A NumPy array of shape (B, H, W, C).
        bboxes (np.ndarray): A NumPy array of shape (B, 4) for bounding boxes,
                             in the format [cx, cy, w, h].
        inp_dim (int): Desired width and height (assumes width = height).

    Returns:
        tuple:
            imgs_tensor (torch.Tensor): A tensor of shape (B, 3, inp_dim, inp_dim).
            original_images (list): List of original images [(H, W, C), ...].
            dims (list): List of (orig_w, orig_h) for each image.
            new_bboxes (np.ndarray): The updated bounding boxes after scaling,
                                     shape (B, 4).
    """
    B = imgs.shape[0]

    # Keep track of original images and their dimensions
    original_images = []
    dims = []
    for i in range(B):
        original_images.append(imgs[i])
        h_, w_ = imgs[i].shape[:2]
        dims.append((w_, h_))

    # Letterbox the entire batch; also update bboxes
    letterboxed, new_bboxes = letterbox_images(imgs, bboxes, (inp_dim, inp_dim))
    # letterboxed shape: (B, inp_dim, inp_dim, 3)

    letterboxed = letterboxed.copy()

    # Convert to float, scale to [0, 1]
    imgs_tensor = torch.from_numpy(letterboxed).float()

    return imgs_tensor, original_images, dims, new_bboxes


def test_transform_torch(patch, shrink_rate):
    """Code adapt from alphapose.utils.presets.simple_transform.py
    
    patch: torch.Tensor, shape (3, H, W)
    shrink_rate: torch.Tensor, shape [N]
    """
    
    # TODO: hard code the ratio here for attacking alpha pose
    _aspect_ratio = 0.75
    _input_size = [256, 192]    # [H, W]
    
    N = shrink_rate.size(0)
    C, H, W = patch.size()
    
    center = torch.tensor([W / 2, H / 2])[None, ].repeat(N, 1)
    randomize_center = center + torch.rand_like(center) * 0.1
    
    w = shrink_rate * H * math.sqrt(_aspect_ratio)
    h = shrink_rate * H / math.sqrt(_aspect_ratio)
    
    crop_box = torch.cat([randomize_center - torch.stack([w, h], dim=1) / 2,
                          randomize_center + torch.stack([w, h], dim=1) / 2], dim=1).long()
    
    # First crop, then resize, the crop and resize should be differentiable
    new_images = torch.zeros(N, C, _input_size[0], _input_size[1]).cuda()
    for i in range(N):
        img = patch[:, crop_box[i][1]:crop_box[i][3], crop_box[i][0]:crop_box[i][2]]
        new_images[i] = F.interpolate(img[None, ], (_input_size[0], _input_size[1]), mode='bilinear', align_corners=False)

    new_images = new_images / 255.0
    new_images[:, 0].add_(-0.406)
    new_images[:, 1].add_(-0.457)
    new_images[:, 2].add_(-0.480)

    return new_images