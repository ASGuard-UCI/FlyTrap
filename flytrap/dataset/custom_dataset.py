import json
import os
from typing import Callable, List, Union

import cv2
import numpy as np
from mmengine.dataset import Compose
from mmengine.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Dataset for training adversarial examples

    TODO:
        [ ] by default use cross search-template pairs

    :args
        meta_file: str, path to the meta file
        pipeline: callable, image transformation
    """

    def __init__(self,
                 meta_file: str,
                 pipeline: List[Union[dict, Callable]] = []):

        self.meta_file = meta_file
        self.parse_data_info(meta_file)

        self.pipeline = Compose(pipeline)

    def parse_data_info(self, meta_file: str = None):

        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        print('Parsing meta file...')
        self.search_imgs = []
        self.search_umbrella_rects = []
        self.template_imgs = []
        self.template_rects = []
        # print('Search videos:')
        for video in self.meta['search']:
            # print(f"{video}: {len(self.meta['search'][video]['imgs'])} frames")
            self.search_imgs += self.meta['search'][video]['imgs']
            self.search_umbrella_rects += self.meta['search'][video]['anno']
        # print('Template videos:')
        for video in self.meta['template']:
            # print(f"{video}: {len(self.meta['template'][video]['imgs'])} frames")
            self.template_imgs += self.meta['template'][video]['imgs']
            self.template_rects += self.meta['template'][video]['anno']
        print(f"Total search images: {len(self.search_imgs)}\nTotal template images: {len(self.template_imgs)}")

    def prepare_train_data(self, index: int):
        # load search image
        input_dict = dict(
            search=self.search_imgs[index],
            umbrella_bbox=np.array(self.search_umbrella_rects[index]),
            video=os.path.basename(os.path.dirname(self.search_imgs[index])),
            all_template=self.template_imgs,
            all_template_bbox=self.template_rects)
        return input_dict

    def __len__(self):
        return len(self.search_imgs)

    def __getitem__(self, index: int):
        # load search image
        input_dict = self.prepare_train_data(index)
        input_dict = self.pipeline(input_dict)

        return input_dict


class VideoDataset(Dataset):
    """Video dataset """

    def __init__(self,
                 images: List[str],
                 anno: List[List[float]],
                 init_bbox: List[float],
                 start_frame: int,
                 pipeline: List[Union[dict, Callable]] = []):
        self.images = images
        self.anno = anno
        self.init_bbox = init_bbox
        self.start_frame = start_frame
        self.pipeline = pipeline
        assert len(images) == len(anno), 'Number of images and annotations should be the same'

    def __len__(self):
        return len(self.images)

    def prepare_test_data(self, index: int):
        output_dict = dict(
            img_path=self.images[index],
            umbrella_bbox=self.anno[index],
            init_bbox=self.init_bbox,
            apply_attack=True if index > self.start_frame else False
        )

        return output_dict

    def __getitem__(self, index: int):
        results = self.prepare_test_data(index)
        results = self.pipeline(results)
        return results


@DATASETS.register_module()
class CustomEvalDataset(Dataset):
    """Dataset for evaluating adversarial examples"""
    """Dataset for training adversarial examples

    TODO:
        [ ] by default use cross search-template pairs

    :args
        meta_file: str, path to the meta file
        pipeline: callable, image transformation
    """

    def __init__(self,
                 meta_file: str,
                 sub_string: str,
                 pipeline: List[Union[dict, Callable]] = []):
        self.meta_file = meta_file
        self.sub_string = sub_string
        self.pipeline = Compose(pipeline)
        self.parse_data_info(meta_file)

    def parse_data_info(self, meta_file: str = None):
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        print('Parsing meta file...')
        self.videos = {}
        videos = list(self.meta.keys())
        videos.sort()
        for video in videos:
            if self.sub_string not in video:
                continue
            self.videos[video] = VideoDataset(
                images=self.meta[video]['img_files'],
                anno=self.meta[video]['anno'],
                init_bbox=self.meta[video]['init_bbox'],
                start_frame=self.meta[video]['start_frame'],
                pipeline=self.pipeline
            )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int):
        raise NotImplementedError("Use the `.videos` attribute to access the video dataset.")
