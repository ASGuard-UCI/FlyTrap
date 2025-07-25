## Physical engine: Adaptive Attack towards PercepGuard

import numpy as np
import cv2
import math
import torch

from .utlis import prep_frames, test_transform_torch
from ..builder import ENGINE, APPLYER


def crop(image, bbox, output_sz):
    """ Used in both distance simulator and crop search area 
    Args:
        image: [H, W, 3]
        bbox: [4] [x1, y1, x2, y2]
        output_sz: (H, W)
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    H_out, W_out = output_sz
    
    avg_padding_value = np.mean(image, axis=(0, 1)).astype(np.uint8).tolist()
    left_padding = np.maximum(0, -x1)
    right_padding = np.maximum(0, x2 - W)
    top_padding = np.maximum(0, -y1)
    bottom_padding = np.maximum(0, y2 - H)
    image_pad = cv2.copyMakeBorder(image, int(top_padding), int(bottom_padding), int(left_padding), int(right_padding), cv2.BORDER_CONSTANT, value=avg_padding_value)
    x1_new, x2_new = x1 + left_padding, x2 + left_padding
    y1_new, y2_new = y1 + top_padding, y2 + top_padding
    cropped_image = cv2.resize(image_pad[int(y1_new):int(y2_new), int(x1_new):int(x2_new)], (W_out, H_out))
    
    return cropped_image


def update_bbox(image, crop_bbox, original_bbox, output_sz):
    """Update the bbox location in the cropped images
    Args:
        image: [H, W, 3]
        crop_bbox: [4] [cx, cy, w, h]
        original_bbox: [4] [cx, cy, w, h]
        output_sz: (H, W)
    Returns:
        bbox: [4] [cx, cy, w, h]"""
    H, W = image.shape[:2]
    cx, cy, crop_width, crop_height = crop_bbox
    # prevent overwrite the original bbox
    original_bbox = original_bbox.copy()
    
    original_bbox[0] = original_bbox[0] - (cx - crop_width / 2)
    original_bbox[1] = original_bbox[1] - (cy - crop_height / 2)
    
    # tracker crop square, so we assume w=h to double check
    assert crop_width == crop_height
    original_bbox[::2] = original_bbox[::2] * output_sz / crop_width
    original_bbox[1::2] = original_bbox[1::2] * output_sz / crop_height
    
    return original_bbox


@ENGINE.register_module()
class PhysicalEnginePercepGuard:
    def __init__(self, 
                 render: dict = None, 
                 applyer: dict = None,
                 history=None, 
                 evolve_step=4, 
                 interval_distance=2, # meter
                 focal=2.4e-2, # meter
                 human_ratio=2,
                 sensor_size = [7.2e-4, 9.6e-4], # [H, W]
                 distance_threshold = [2, 4, 8],
                 shrink_rate = [0.05, 0.1, 0.2, 0.4],
                 search_factor=4.5,
                 size=0.4,
                 out_size=320,
                 det_out_size=608):
        """
        Modification:
            1.  Based on V2, Don't use distance-based renderer
            2.  In the first frame, use random factor strategy in CS design
            3.  Prepare the adv image and target bbox loc for object detections (e.g. YOLO) for adaptive attack
        Args:
            render: 
            history: TODO
            evolve_step (int): number of steps to evolve the physical system
            focal (float): focal length of the camera, default is 24 for DJI Mini 4 Pro (meter)
            obj_human_ratio (float): ratio of the object size to human size
            human_ratio (float): human height / human width, adaptive attack to perceiveguard by optimize the human ratio
            sensor_size (List[float]): size of the camera sensor [H, W] (meter)
            size (float): size of the physical umbrella radius (meter)
            search_factor (float): search factor for the search area
            out_size (int): output size of the search region, depends on tne input resolution of tracking model
        """
        # I think the `eolve_step` and `interval_distance` are the key parameters, 
        # influence the effectivenss of the attack. 
        # If it's both too small, the attack will lock on the umbrella,
        # causing slightly pulling effects.
        # I think there is a trade-off between attack effectiveness and attack smoothness.
        # If we consider spatial-temporal consistency
        self.evolve_step = evolve_step
        self.interval_distance = interval_distance
        # assume the human is 1.8m tall and 0.41m wide
        self.obj_human_ratio = (size * 2)**2 / (1.8 * 0.41)
        self.human_ratio = human_ratio
        self.sensor_size = sensor_size
        self.size = size
        self.focal = focal
        self.rectify_scale = 13.
        # TODO: manually set the distance threshold for continuous shrinking
        self.distance_threshold = distance_threshold
        # shrink rate is compared with the original size of the umbrella
        self.shrink_rate = shrink_rate
        self.render = APPLYER.build(render)
        self.applyer = APPLYER.build(applyer)
        # TODO: we currently use a fixed value based on MixFormer config
        self.search_factor = search_factor
        self.out_size = out_size
        self.det_out_size = det_out_size
    
    def distance_estimate(self, bbox, H, W):
        """Estimate the distance of the object from the camera
        Args:
            bbox (List[np.ndarray]): list of bounding boxes [4]
        Returns:
            distance (List[float]): distance of the object from the camera
        """
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        img_area = w * h
        physical_area = self.size * self.size
        pixel_height, pixel_width = self.sensor_size[0] / H, self.sensor_size[1] / W
        ## TODO: currently manually set the scale factor, recalculate the scale factor
        distance = np.sqrt(physical_area / (img_area * pixel_height * pixel_width * self.rectify_scale**2)) * self.focal
        return distance
        
    def distance_to_bbox(self, distance, H, W):
        """Convert distance to bounding box, assume w=h for umbrellas
        Args:
            distance (List[float]): distance of the object from the camera
        Returns:
            bbox (List[np.ndarray]): list of bounding boxes [4]
        """
        pixel_height, pixel_width = self.sensor_size[0] / H, self.sensor_size[1] / W
        physical_area = self.size * self.size
        w = np.sqrt(physical_area) * self.focal / (distance * np.sqrt(pixel_height * pixel_width * self.rectify_scale**2))
        return w        
    
    def distance_simulater(self, image, bbox, target_bbox, H, W):
        """Crop the image, the cropped ratio is the same as the original image
        After crop, resize the image to original size, the bbox after resize should equal to target_bbox
        Args:
            image: [B, H, W, 3]
            bbox: [B, 4] [cx, cy, w, h]
            target_bbox: [self.evolve_step, B] (float) assume w=h for umbrellas
        Returns:
            cropped_images: [B, self.evolve_step, H, W, 3]
        """
        B = image.shape[0]
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3] # [B, ]
        cropped_images = np.zeros((B, self.evolve_step - 1, H, W, 3))
        crop_height = H * target_bbox[0] / target_bbox[1:] # [self.evolve_step - 1, B]
        crop_width = W * target_bbox[0] / target_bbox[1:] # [self.evolve_step - 1, B]
        
        x1 = cx[None, ] - crop_width / 2
        y1 = cy[None, ] - crop_height / 2
        x2 = cx[None, ] + crop_width / 2
        y2 = cy[None, ] + crop_height / 2
        
        for i in range(self.evolve_step - 1):
            for j in range(B):
                cropped_images[j, i] = crop(image[j], np.array([x1[i, j], y1[i, j], x2[i, j], y2[i, j]]), (H, W))
        cropped_images = np.concatenate([image[:, None], cropped_images], axis=1)
        return cropped_images

    def x1y1wh_to_cxcywh(self, bbox):
        """Convert bbox from x1y1wh to cxcywh
        Args:
            bbox: [B, 4] [x1, y1, w, h]
        Returns:
            bbox: [B, 4] [cx, cy, w, h]
        """
        cx = bbox[:, 0] + bbox[:, 2] / 2
        cy = bbox[:, 1] + bbox[:, 3] / 2
        return np.stack([cx, cy, bbox[:, 2], bbox[:, 3]], axis=1)

    def adv_target_bbox_generator(self, bbox, distance):
        """Generate the adversarial target bbox based on the distance
        Args:
            bbox: [self.evolve_step, B] (float) assume w=h for umbrellas
            distance: [self.evolve_step, B] (float)
            """
            
        # generate the target bbox based on the distance threshold
        # the shrink rate and threshold are predefined manually
        # TODO: can we optimize here?
        # categorize the distance into different stages based on the threshold
        stage = np.digitize(distance, self.distance_threshold) # [self.evolve_step, B]
        
        # determine the shrink rate based on the stage
        shrink_rate = np.take(self.shrink_rate, stage) # [self.evolve_step, B]
        
        # determine the adversarial target bbox size based on the shrink rate
        adv_target_bbox_size = bbox * shrink_rate
        
        # mimic human ratio for spatial consistency
        # but human shape the height might be larger than the umbrella radius
        adv_target_bbox_height = adv_target_bbox_size * np.sqrt(self.human_ratio)
        adv_target_bbox_width = adv_target_bbox_size / np.sqrt(self.human_ratio)
        
        # If height exceeds bbox, scale both dimensions proportionally to maintain ratio
        height_exceeds = adv_target_bbox_height > bbox
        scale_factor = np.where(height_exceeds, bbox / adv_target_bbox_height, 1.0)
        adv_target_bbox_height = adv_target_bbox_height * scale_factor
        adv_target_bbox_width = adv_target_bbox_width * scale_factor
        
        # Final check to ensure both dimensions are within bounds
        adv_target_bbox_height = np.minimum(adv_target_bbox_height, bbox)
        adv_target_bbox_width = np.minimum(adv_target_bbox_width, bbox)
        assert np.all(adv_target_bbox_height <= bbox)
        assert np.all(adv_target_bbox_width <= bbox)
        assert np.allclose(adv_target_bbox_height / adv_target_bbox_width, self.human_ratio, rtol=1e-3)
        return np.stack([adv_target_bbox_width, adv_target_bbox_height], axis=2) # [self.evolve_step, B, 2] [w, h]

    def crop_search_area(self, image, umbrella_bbox, target_bbox, H, W):
        """Crop the search area based
        
        In this function, we consider the tracker dynamics, the search area
        is based on the target adversarial bbox in the previous closed-loop frame 
        
        for the frist frame, the search area is estimated based on the estimation of human size
        
        during crop, we randomize the bbox location while making sure the target_bbox is within the cropped image
        to improve the attack robustness
        
        Args:
            image: [self.evolve_step, B, H, W, 3]
            umbrella_bbox: [self.self.evolve_step, B, 4] [cx, cy, w, h]
            target_bbox: [self.evolve_step, B, 4] [cx, cy, w, h]
        """
        
        def _randomize_bbox(state_bbox, target_bbox, factor):
            """Randomize the target_bb location in cropped image,
            if not, the target_bb will always at the center of the cropped image
            
            Args:
                state_bbox: [self.self.evolve_step, B, 4] [cx, cy, w, h]
                target_bbox: [self.evolve_step, B, 4] [cx, cy, w, h]
            Returns:
                crop_bbox: [self.evolve_step, B, 4] [cx, cy, w, h] the bbox to crop the image as search region
            """
            evolve_step, B = state_bbox.shape[:2]
            cx_s, cy_s, w_s, h_s = state_bbox[:, :, 0], state_bbox[:, :, 1], state_bbox[:, :, 2], state_bbox[:, :, 3]
            cx_t, cy_t, w_t, h_t = target_bbox[:, :, 0], target_bbox[:, :, 1], target_bbox[:, :, 2], target_bbox[:, :, 3]
            # size of the cropped image
            # TODO: double check here, if factor < 1, the shift_x and shift_y will be negative
            crop_sz = np.ceil(np.sqrt(w_s * h_s) * factor)
            shift_x = (crop_sz - w_t) // 2 - 1
            shift_y = (crop_sz - h_t) // 2 - 1
            x_new = cx_t + np.random.uniform(-shift_x, shift_x, size=(evolve_step, B))
            y_new = cy_t + np.random.uniform(-shift_y, shift_y, size=(evolve_step, B))
            return np.stack([x_new, y_new, crop_sz, crop_sz], axis=2)
        
        # `state_bbox` is the tracker predict bbox in the last frame
        # in the first frame, we estimated based on the human size
        # in the simulated frame, we use the adversarial target bbox in the last frame
        
        # since the search area only depends on the bbox size, not ratio
        # we don't need to consider the human ratio here
        # TODO: the logic here is not clear, need to be optimized
        #       in the `distance_simulator`, we manually set the umbrella center to be the center of the image
        #       this might cause problems in search center inconsistency between t0 and t1
        
        B = image.shape[1]
        # prevent overwriting
        target_bbox = target_bbox.copy()
        
        first_frame_random_search_factor = np.random.uniform(1, 6)
        crop_bbox_first_frame = _randomize_bbox(umbrella_bbox[0:1], target_bbox[0:1], first_frame_random_search_factor)
        
        state_bbox = target_bbox[:-1].copy() # [self.evolve_step - 1, B, 4]
        # TODO: word around, manually set t1 bbox to be the center
        state_bbox[0, :, :2] = np.array([W / 2, H / 2])
        # randomize the bbox location in the cropped image, ensure
        #   1. the search area size is dependent on the state_box
        #   2. the target_bbox is within the cropped image
        crop_bbox = _randomize_bbox(state_bbox, target_bbox[1:], self.search_factor) # [self.evolve_step - 1, B, 4] [cx, cy, w, h]

        crop_bbox = np.concatenate((crop_bbox_first_frame, crop_bbox), axis=0) # [self.evolve_step, B, 4] [cx, cy, w, h]

        umbrella_bbox_new = np.zeros((self.evolve_step, B, 4))
        target_bbox_new = np.zeros((self.evolve_step, B, 4))
        input_image = np.zeros((self.evolve_step, B, self.out_size, self.out_size, 3))
        for i in range(self.evolve_step):
            for j in range(B):
                crop_bbox_x1y1x2y2 = np.array([crop_bbox[i, j][0] - crop_bbox[i, j][2] / 2, 
                                               crop_bbox[i, j][1] - crop_bbox[i, j][3] / 2, 
                                               crop_bbox[i, j][0] + crop_bbox[i, j][2] / 2, 
                                               crop_bbox[i, j][1] + crop_bbox[i, j][3] / 2])
                # crop use x1y1x2y2
                input_image[i, j] = crop(image[i, j], crop_bbox_x1y1x2y2, (self.out_size, self.out_size))
                # update box use cx, cy, w, h
                # TODO: make it more consistent
                umbrella_bbox_new[i, j] = update_bbox(image[i, j], crop_bbox[i, j], umbrella_bbox[i, j], self.out_size)
                target_bbox_new[i, j] = update_bbox(image[i, j], crop_bbox[i, j], target_bbox[i, j], self.out_size)
        return input_image, umbrella_bbox_new, target_bbox_new

    def distance_renderer(self, patch, target_distance):
        B = target_distance.shape[1]
        rendered_patch = self.render(patch)

        return rendered_patch

    def distance_applyer(self, images, bbox, patch):
        """
        Args:
            images: [B, self.evolve_step, H, W, 3]
            bbox: [self.evolve_step, B, 4] [cx, cy, w, h]
            patch: List [self.evolve_step * B]
        """
        def normalize(bbox, H, W):
            """Normalize the bbox to [-1, 1] for coordinate transformation
            Args:
                bbox: [self.evolve_step, B, 4] [cx, cy, w, h]
            """
            cx, cy, w, h = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
            coords = np.stack(  # [self.evolve_step, B, 4, 2]
                [
                    np.stack([cx - w / 2, cy - h / 2], axis=-1),
                    np.stack([cx + w / 2, cy - h / 2], axis=-1),
                    np.stack([cx + w / 2, cy + h / 2], axis=-1),
                    np.stack([cx - w / 2, cy + h / 2], axis=-1)
                ],
                axis=2
            )

            # norm to [-1, 1]
            # used for apply the patch
            coords = (coords - np.array([W / 2, H / 2])) / np.array([W / 2, H / 2])
            return coords

        evolve_step, B, H, W = images.shape[:4]
        normalized_bbox = torch.tensor(normalize(bbox, H, W)).cuda()
        
        composed_images = torch.zeros((evolve_step, B, H, W, 3)).cuda()
        for e in range(evolve_step):
            for b in range(B):
                image_torch = torch.tensor(images[e, b], dtype=torch.float32).unsqueeze(0).cuda()
                composed_image_, _ = self.applyer(image_torch, patch, normalized_bbox[e, b][None, ])
                composed_images[e, b] = composed_image_.squeeze()
        
        return composed_images

    def prepare_det_input(self, images, bbox, H, W):
        """Prepare the input for object detection model, update the bbox for patch applyer
        """
        B = images.shape[0]
        det_input_images, _, _, new_bbox = prep_frames(images, bbox, self.det_out_size)
        # shape: [B, 4]
        orig_shape = np.array([W, H, W, H])[None, ].repeat(B, axis=0)
        return det_input_images, new_bbox, orig_shape

    def prepare_pose_input(self):
        pass

    def __call__(self, images, bbox, patch):
        """Run the physical engine to generate the simulation
        Args:
            images: [B, H, W, 3]
            bbox : [B, 4] [x1, y1, w, h]
            patch (np.ndarray): [3, H, W]
        """
        
        # TODO: the dataloader convert numpy to tensor:cuda, here we convert it back
        # This is low efficiency, need to be optimized
        
        images = images.cpu().numpy()
        B, H, W = images.shape[:3]
        bbox = self.x1y1wh_to_cxcywh(bbox.cpu().numpy())
        
        # estimate the initial distance
        distances = self.distance_estimate(bbox, H, W)  # [B, ]
        # create target distance based on interval_distance and evolve_step
        target_distance = np.linspace(distances, 
                                      distances - (self.evolve_step - 1) * self.interval_distance, 
                                      self.evolve_step)
        # clamp the distance to be positive
        target_distance = np.clip(target_distance, 0.5, None) # [self.evolve_step, B]
        
        ## Be careful with the dimension order here!
        
        # reverse target bbox size based on target_distance
        # represent umbrella bbox size after moving closer to the target_distance (e.g., be larger)
        target_bbox_size = self.distance_to_bbox(target_distance, H, W) # [self.evolve_step, B]
        
        # this step simulate the drone moving closer by zooming the image, thus the current bbox size should be the target_bbox_size
        # after zoom in, the umbrella center overlap with the image center
        # the final image size is the same as the original image size
        distance_images = self.distance_simulater(images, bbox, target_bbox_size, H, W) # [B, self.evolve_step, H, W, 3]
        
        # this step compute the adversarial target bbox for each step to based on distance stage threshold
        # The `target_shrink_bbox` is the absolute coordinate in the original image
        target_shrink_bbox = self.adv_target_bbox_generator(target_bbox_size, target_distance) # [self.evolve_step, B, 2] [h, w]
        
        # intergrate the new umbrella bbox => [self.evolve_step, B, 4] [cx, cy, w, h]
        umbrella_center = bbox[None, :, :2] # [1, B, 2]
        # center of `distance_images` is always the center of the image
        evolve_umbrella_center = np.ones((self.evolve_step - 1, B, 2)) * np.array((W / 2, H / 2)) # [self.evolve_step - 1, B, 2]
        umbrella_center = np.concatenate((umbrella_center, evolve_umbrella_center), axis=0) # [self.evolve_step, B, 2]
        # assume w=h for umbrellas
        target_bbox_size = np.stack([target_bbox_size, target_bbox_size], axis=2) # [self.evolve_step, B, 2] [w, h]
        # clamp the target_bbox_size to be within the image size
        target_bbox_size = np.minimum(target_bbox_size, np.array([W, H]))
        umbrella_bbox = np.concatenate((umbrella_center, target_bbox_size), axis=2) # [self.evolve_step, B, 4] [cx, cy, w, h]
        target_shrink_bbox = np.concatenate((umbrella_center, target_shrink_bbox), axis=2)
        # NOTE: the `input_image` dimension is [self.evolve_step, B, H, W, 3]
        input_image, umbrella_bbox_new, target_bbox_new = self.crop_search_area(distance_images.transpose(1, 0, 2, 3, 4), umbrella_bbox, target_shrink_bbox, H, W)
        
        # TODO: prepare input for the pose estimation model
        # pose_input_image, pose_target_ht = self.prepare_pose_input(input_image, target_bbox_new, H, W)
        
        # debug
        # debug_bbox1 = umbrella_bbox[0, 0].copy()
        # debug_bbox2 = target_shrink_bbox[0, 0].copy()
        # debug_img = distance_images[0, 0].copy()
        # debug_img = cv2.rectangle(debug_img, 
        #                           (int(debug_bbox1[0] - debug_bbox1[2] / 2), 
        #                            int(debug_bbox1[1] - debug_bbox1[3] / 2)),
        #                            (int(debug_bbox1[0] + debug_bbox1[2] / 2), 
        #                            int(debug_bbox1[1] + debug_bbox1[3] / 2)), 
        #                           (0, 255, 0), 2)
        # debug_img = cv2.rectangle(debug_img,
        #                             (int(debug_bbox2[0] - debug_bbox2[2] / 2), 
        #                              int(debug_bbox2[1] - debug_bbox2[3] / 2)), 
        #                              (int(debug_bbox2[0] + debug_bbox2[2] / 2), 
        #                              int(debug_bbox2[1] + debug_bbox2[3] / 2)), 
        #                             (255, 0, 0), 2)
        # cv2.imwrite('debug_bbox.png', debug_img)
        
        # render the patch on the umbrella, then compose with the umbrella position in the image
        # the render process use the estimated distance

        rendered_patch = self.distance_renderer(patch, target_distance)
        composed_images = self.distance_applyer(input_image, umbrella_bbox_new, rendered_patch) # [self.evolve_step, B, H, W, 3]
        
        # normalize the bbox to [0, 1]
        target_bbox_new = target_bbox_new / self.out_size
        umbrella_bbox_new = umbrella_bbox_new / self.out_size
        
        target_bbox_new = torch.tensor(target_bbox_new).cuda() # [self.evolve_step, B, 4] [cx, cy, w, h]
        umbrella_bbox_new = torch.tensor(umbrella_bbox_new).cuda() # [self.evolve_step, B, 4] [cx, cy, w, h]
        
        return composed_images, umbrella_bbox_new, target_bbox_new