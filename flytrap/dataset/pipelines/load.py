import math
import warnings
from typing import Optional, List, Union

import cv2 as cv
import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
from mmcv.transforms import TRANSFORMS, BaseTransform

from ...builder import PIPELINES
from ...utils import box_xywh_to_xyxy


@TRANSFORMS.register_module()
class CustomLoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 img_keys: List[str] = ['img_path'],
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.img_keys = img_keys
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load(self, filename):
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image. Support load list of images

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        for key in self.img_keys:

            if key not in results:
                raise KeyError(f'{key} is required in results')

            filename = results[key]
            if isinstance(filename, List):
                img = [self._load(f) for f in filename]
                shape = img[0].shape[:2]
            else:
                img = self._load(filename)
                shape = img.shape[:2]

            results[key] = img
            results['img_shape_' + key] = shape
            results['ori_shape_' + key] = shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@PIPELINES.register_module()
class NormalizeCoordinates:
    """Get normalized coordinates of the umbrella bbox
        Used for apply the patch"""

    def __call__(self, results):
        bbox = results['umbrella_bbox']
        # TODO: [] hard code here
        if 'img_shape' in results:
            height, width = results['img_shape']
        elif 'img_shape_search' in results:
            height, width = results['img_shape_search']
        else:
            raise ValueError("Need img_shape or img_shape_search")

        x, y, w, h = bbox
        coords = np.array(  # [4, 2]
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]]
        )

        # norm to [-1, 1]
        # used for apply the patch
        coords = (coords - np.array([width / 2, height / 2])) / np.array([width / 2, height / 2])
        results['coords'] = coords
        # original dataset bbox: [x1, y1, w, h]
        # used for mixformer score head
        normalize_bbox = box_xywh_to_xyxy(bbox)
        normalize_bbox = (normalize_bbox / np.array([width, height, width, height])).astype(np.float32)
        results['normalize_bbox'] = normalize_bbox
        return results


@PIPELINES.register_module()
class TemplateSample:
    def __init__(self,
                 same_video: bool = False,
                 online_template: bool = False,
                 num_online: int = 5,
                 rand_blend: bool = False):
        """Sample a template frames
        Args:
            same_video: whether sample from the same video
            online_template: whether sample online template
            num_online: number of online template frames
            rand_blend: apply image blending for data augmentation
        TODO:
         - [x] support same video
         - [] support multiple online template
        """
        self.same_video = same_video
        self.online_template = online_template
        self.num_online = num_online
        self.rand_blend = rand_blend
        if self.online_template:
            assert num_online < 2, "Now only support 1 online template frames"

    def __call__(self, results):
        all_template = results['all_template']
        all_template_bbox = results['all_template_bbox']
        video = results['video']
        # sample template from the same video
        if self.same_video:
            all_template, all_template_bbox = self.filter_template(video, all_template, all_template_bbox)
        # add online template for MixFormer
        sample_num = 1 if not self.online_template else 1 + self.num_online
        # add one more for data augmentation
        sample_num = sample_num + 1 if self.rand_blend else sample_num
        
        sample_idx = np.random.choice(len(all_template), sample_num, replace=False)
        template_img = all_template[sample_idx[0]]
        template_bbox = all_template_bbox[sample_idx[0]]
        results['template'] = template_img
        results['template_bbox'] = template_bbox
        if self.online_template:
            # online_template_img = [all_template[i] for i in sample_idx[1:]]
            # online_template_bbox = [all_template_bbox[i] for i in sample_idx[1:]]
            # results['online_template'] = online_template_img
            # results['online_template_bbox'] = online_template_bbox
            online_template_img = all_template[sample_idx[1]]
            online_template_bbox = all_template_bbox[sample_idx[1]]
            results['online_template'] = online_template_img
            results['online_template_bbox'] = online_template_bbox
        
        if self.rand_blend:
            blend_img = all_template[sample_idx[2]]
            blend_bbox = all_template_bbox[sample_idx[2]]
            results['blend_template'] = blend_img
            results['blend_template_bbox'] = blend_bbox
        
        return results

    def filter_template(self, video, all_template, all_template_bbox):
        items = [item for item in zip(all_template, all_template_bbox) if video in item[0]]
        all_template = [item[0] for item in items]
        all_template_bbox = [item[1] for item in items]
        return all_template, all_template_bbox


@PIPELINES.register_module()
class CropTargetObject:
    """Crop Target Object according to the bbox"""

    def __init__(self,
                 template_factor: float = 2.0,
                 search_factor: Union[float, List] = 4.5,
                 template_size: int = 128,
                 search_size: int = 320,
                 randomize: bool = True):
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.template_size = template_size
        self.search_size = search_size
        self.randomize = randomize
        if randomize:
            assert isinstance(search_factor, List), "search_factor should be a list"
            assert len(search_factor) == 2, "search_factor should be a list of length 2"
            assert search_factor[0] < search_factor[1], "search_factor should be a list of length 2"

    def _randomize_bbox(self, bbox, factor, h, w):
        """Randomize the target_bb location in cropped image,
        if not, the target_bb will always at the center of the cropped image"""
        if not isinstance(bbox, list):
            x, y, w, h = bbox.tolist()
        else:
            x, y, w, h = bbox
        # size of the cropped image
        crop_sz = math.ceil(math.sqrt(w * h) * factor)
        shift_x = (crop_sz - w) // 2 - 1
        shift_y = (crop_sz - h) // 2 - 1
        x_new = x + np.random.uniform(-shift_x, shift_x)
        y_new = y + np.random.uniform(-shift_y, shift_y)
        return [x_new, y_new, w, h]

    def _call_single(self, im, center_bb, target_bb, search_area_factor, output_sz=None):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            center_bb - center of the cropped image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(center_bb, list):
            x, y, w, h = center_bb.tolist()
        else:
            x, y, w, h = center_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

        # Calculate new bounding box coordinates
        x_tar, y_tar, w_tar, h_tar = target_bb
        new_x = x_tar - x1
        new_y = y_tar - y1
        new_w = w_tar
        new_h = h_tar
        new_target_bb = np.array([new_x, new_y, new_w, new_h])

        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
            new_target_bb = (new_target_bb * resize_factor).astype(np.int32)

        # debug
        # img = cv.rectangle(im_crop_padded.astype(np.uint8), (new_target_bb[0], new_target_bb[1]),
        #              (new_target_bb[0] + new_target_bb[2], new_target_bb[1] + new_target_bb[3]), (0, 255, 0))
        # cv.imwrite('debug.jpg', img)
        return im_crop_padded, new_target_bb, (output_sz, output_sz)

    def __call__(self, results):
        # TODO: [] Hard code keys here
        assert 'search' in results.keys(), "Need search image"
        assert 'template' in results.keys(), "Need template image"

        # search image
        img = results['search']
        h, w = img.shape[:2]
        bbox = results['umbrella_bbox']
        if self.randomize:
            # random shift the bbox and factor
            search_factor = np.random.uniform(self.search_factor[0], self.search_factor[1])
            center_bbox = self._randomize_bbox(bbox, search_factor, h, w)
        else:
            search_factor = self.search_factor
            center_bbox = bbox
        new_img, new_bbox, new_shape = self._call_single(img, center_bbox, bbox, search_factor, self.search_size)
        results['search'] = new_img
        results['umbrella_bbox'] = new_bbox
        results['img_shape_search'] = new_shape

        # template image
        img = results['template']
        bbox = results['template_bbox']
        new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
        results['template'] = new_img
        results['template_bbox'] = new_bbox
        results['img_shape_template'] = new_shape

        # multiple online template: not support now
        # if 'online_template' in results.keys():
        #     imgs = results['online_template']
        #     bboxes = results['online_template_bbox']
        #     new_imgs = []
        #     new_bboxes = []
        #     for img, bbox in zip(imgs, bboxes):
        #         new_img, new_bbox, new_shape = self._call_single(img, bbox, self.template_factor, self.template_size)
        #         new_imgs.append(new_img)
        #         new_bboxes.append(new_bbox)
        #     results['online_template'] = new_imgs
        #     results['online_template_bbox'] = new_bboxes
        #     results['img_shape_online_template'] = new_shape

        # TODO: [] hard code check here
        if 'online_template' in results.keys():
            img = results['online_template']
            bbox = results['online_template_bbox']
            new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
            results['online_template'] = new_img
            results['online_template_bbox'] = new_bbox

        if 'blend_template' in results.keys():
            img = results['blend_template']
            bbox = results['blend_template_bbox']
            new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
            results['blend_template'] = new_img
            results['blend_template_bbox'] = new_bbox

        return results


@PIPELINES.register_module()
class CustomNormalize:
    def __init__(self,
                 img_keys: List[str],
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 normalize_factor: float = 255.0):
        self.img_keys = img_keys
        self.mean = torch.Tensor(mean).view(1, 1, 3)
        self.std = torch.Tensor(std).view(1, 1, 3)
        # just to ensure compatibility with previous code
        # previous code hard coded the normalization factor to 255
        # so we keep it as default
        self.normalize_factor = normalize_factor

    def __call__(self, results):
        for key in self.img_keys:
            img = results[key]
            if isinstance(img, List):
                assert isinstance(img[0], torch.Tensor), "Now only support list of torch.Tensor"
                img_norm = []
                for i in img:
                    device = i.device
                    img_norm.append(
                        (((i / self.normalize_factor) - self.mean.to(device)) / self.std.to(device)).permute(0, 3, 1,
                                                                                                             2))
            else:
                assert isinstance(img, torch.Tensor), "Now only support torch.Tensor"
                device = img.device
                img_norm = (((img / self.normalize_factor) - self.mean.to(device)) / self.std.to(device)).permute(0, 3,
                                                                                                                  1, 2)
            results[key] = img_norm
        return results


@PIPELINES.register_module()
class CustomCollect:
    def __init__(self, keys: List[str], meta_keys: List[str]):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        data['meta'] = {}
        for key in self.keys:
            data[key] = results[key]
        for key in self.meta_keys:
            data['meta'][key] = results[key]
        return data
    
    
@PIPELINES.register_module()
class BlendImage:
    def __init__(self, alpha: float = 0.5, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, results):
        assert 'blend_template' in results.keys(), "Need blend_template image"
        assert 'template' in results.keys(), "Need template image"
        if np.random.rand() < self.prob:
            blend_img = results['blend_template']
            template_img = results['template']
            results['template'] = cv.addWeighted(template_img, self.alpha, blend_img, 1 - self.alpha, 0)
            return results
        else:
            return results
        
        
@PIPELINES.register_module()
class CropTargetObjectTemplate:
    """Crop Target Object according to the bbox, only for template image"""

    def __init__(self,
                 template_factor: float = 2.0,
                 template_size: int = 128):
        self.template_factor = template_factor
        self.template_size = template_size

    def _call_single(self, im, center_bb, target_bb, search_area_factor, output_sz=None):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            center_bb - center of the cropped image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(center_bb, list):
            x, y, w, h = center_bb.tolist()
        else:
            x, y, w, h = center_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

        # Calculate new bounding box coordinates
        x_tar, y_tar, w_tar, h_tar = target_bb
        new_x = x_tar - x1
        new_y = y_tar - y1
        new_w = w_tar
        new_h = h_tar
        new_target_bb = np.array([new_x, new_y, new_w, new_h])

        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
            new_target_bb = (new_target_bb * resize_factor).astype(np.int32)

        return im_crop_padded, new_target_bb, (output_sz, output_sz)

    def __call__(self, results):
        assert 'template' in results.keys(), "Need template image"

        # template image
        img = results['template']
        bbox = results['template_bbox']
        new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
        results['template'] = new_img
        results['template_bbox'] = new_bbox
        results['img_shape_template'] = new_shape

        # multiple online template: not support now
        # if 'online_template' in results.keys():
        #     imgs = results['online_template']
        #     bboxes = results['online_template_bbox']
        #     new_imgs = []
        #     new_bboxes = []
        #     for img, bbox in zip(imgs, bboxes):
        #         new_img, new_bbox, new_shape = self._call_single(img, bbox, self.template_factor, self.template_size)
        #         new_imgs.append(new_img)
        #         new_bboxes.append(new_bbox)
        #     results['online_template'] = new_imgs
        #     results['online_template_bbox'] = new_bboxes
        #     results['img_shape_online_template'] = new_shape

        # TODO: [] hard code check here
        if 'online_template' in results.keys():
            img = results['online_template']
            bbox = results['online_template_bbox']
            new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
            results['online_template'] = new_img
            results['online_template_bbox'] = new_bbox

        if 'blend_template' in results.keys():
            img = results['blend_template']
            bbox = results['blend_template_bbox']
            new_img, new_bbox, new_shape = self._call_single(img, bbox, bbox, self.template_factor, self.template_size)
            results['blend_template'] = new_img
            results['blend_template_bbox'] = new_bbox

        return results


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_frame(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    # img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = img.transpose((2, 0, 1)).copy()
    img_ = img_.astype(np.float32) / 255.0
    return img_, orig_im, dim
    
    
@PIPELINES.register_module()
class DetectionPreprocess:
    """Prepare image for inputting to the neural network.
    use YOLO-v3 input resolution: 608x608 by default
    """
    def __init__(self, output_size: int = 608):
        self.output_size = output_size
        
    def __call__(self, results):
        assert 'search' in results.keys(), "Need img image"
        img = results['search']
        img, orig_im, dim = prep_frame(img, self.output_size)
        results['det_img'] = img
        W, H = dim
        results['ori_shape'] = np.array((W, H, W, H))
        return results