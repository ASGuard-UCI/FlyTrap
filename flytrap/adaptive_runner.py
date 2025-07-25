## The runner extends based runner to support adaptive attacks

import json
import os.path
import time
from collections import ChainMap
from typing import Callable, Mapping, List

import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import wandb
from tqdm import tqdm

import flytrap.builder as builder
import flytrap.utils as utils
from flytrap.attacks.utils import clip_circle_inplace, clip_pixel_inplace


class AdversarialPatchRunner:
    ## Please change this manually when debug MixFormer and SiamRPN
    ## MixFormer
    IMG_CONFIG = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}
    ## SiamRPN
    # IMG_CONFIG = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    def __init__(self,
                 model,     # tracker 
                 detector,  # detector
                 pose_estimator, # pose estimator
                 tracker,
                 applyer,
                 renderer,
                 work_dir: str,
                 config: dict,
                 device: str,
                 epochs: int,
                 patch_path: str,
                 patch: torch.Tensor,
                 patch_transform: Callable,
                 post_transform: Callable,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 train_loader: data.DataLoader,
                 test_loader: data.Dataset,
                 eval_metric: Callable,
                 loss: Callable,
                 log: bool,
                 defense: Callable = None,
                 eval_interval: int = 1,
                 log_cfg: dict = None,
                 debug: bool = False):
        self.model = model  # used for training
        self.detector = detector   # used for detection adaptive attack
        self.pose_estimator = pose_estimator # used for pose adaptive attack
        self.tracker = tracker  # tracker is warp of model, used for evaluation
        self.applyer = applyer
        self.renderer = renderer
        self.work_dir = work_dir
        self.config = config
        self.patch = patch
        self.patch_transform = patch_transform
        self.post_transform = post_transform
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eval_interval = eval_interval
        self.eval_metric = eval_metric
        self.loss = loss
        self.device = device
        self.epochs = epochs
        self.patch_path = os.path.join(work_dir, os.path.basename(patch_path))
        self.log = log
        self.debug = debug
        if log:
            self._init_logging(log_cfg)

        self.patch = self.patch.to(self.device)
        # build optimizer after moving patch to device
        self.optimizer = builder.OPTIMIZER.build(optimizer, self.patch)
        self.scheduler = builder.SCHEDULER.build(scheduler, self.optimizer)
        self.model = self.model.to(self.device)
        self.detector = self.detector.to(self.device)
        # defense
        if defense is not None:
            self.defense = builder.TRANSFORMS.build(defense)
        else:
            self.defense = None
        
        # train adversarial patch
        # always in eval mode
        self.model.eval()
        self.detector.eval()

    def _init_logging(self, cfg):
        """Initialize logging settings."""
        config = dict(self.config)
        del config['log']
        del config['log_cfg']
        wandb.init(**cfg, config=config)
        self.log_step = 0

    def _log(self, metrics: dict, epoch: int = None):
        """Log metrics."""
        if self.log:
            metrics['epoch'] = epoch
            wandb.log(metrics, step=self.log_step)
            self.log_step += 1

    def _save_patch(self, epoch):
        """Save the adversarial patch to a file."""
        save_patch = self.patch.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # TODO: [] hard code interval, keep it the same with eval_interval
        if epoch % 20 == 0:
            success = cv2.imwrite(self.patch_path.replace('.png', f'_epoch{epoch}.png'), save_patch)
        success = cv2.imwrite(self.patch_path, save_patch)
        print(f'Saved patch to {self.patch_path}: success={success}')

    def _to_device(self, data):
        """Move data to device."""
        if isinstance(data, Mapping):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, List):
            return [self._to_device(d) for d in data]
        elif isinstance(data, tuple):
            return tuple(self._to_device(d) for d in data)
        elif isinstance(data, str) or isinstance(data, int) or isinstance(data, float):
            return data
        else:
            raise NotImplementedError

    def train_epoch(self, epoch: int):
        self.patch.requires_grad_()
        cur_time = time.time()
        with tqdm(total=len(self.train_loader), desc="Training") as pbar:
            for batch_idx, data in enumerate(self.train_loader):
                data_time, cur_time = time.time() - cur_time, time.time()
                aug_patch = self.patch_transform(self.patch / 255.0) * 255.0
                # make sure the patch still in circle shape after augmentation
                clip_circle_inplace(aug_patch) 
                data = self._to_device(data)
                # TODO: [] delete theta return here
                # replace search with adversarial patch version
                data['search'], _ = self.applyer(data['search'], self.renderer(aug_patch), coords=data['coords'])
                tracker_data = self.post_transform(data)
                detector_data = self.det_post_transform(data)
                applyer_time, cur_time = time.time() - cur_time, time.time()
                tracker_input_dict = dict(
                    template=tracker_data['template'],
                    # TODO: [] hard code check here
                    online_template=tracker_data['online_template'] if 'online_template' in tracker_data else None,
                    search=tracker_data['search'],
                    run_score_head=True,
                )
                out = self.model(**tracker_input_dict)
                det_out = self.detector(**detector_data)
                model_time, cur_time = time.time() - cur_time, time.time()
                if self.debug:
                    self._debug(out, tracker_data)
                loss_input_dict = dict(
                    # mixformer: [1, 4]
                    # siamrpn: [B, 4, A*H*W],
                    pred_bbox=out[0]['pred_boxes'].squeeze(),  # [cx, cy, w, h], [0, 1]
                    # mixformer: float [0, 1]
                    # siamrpn: tensor [B, A*H*W]
                    pred_score=out[0]['pred_scores'],
                    gt_bbox=utils.box_xyxy_to_cxcywh_tensor(data['normalize_bbox']),
                    patch=self.patch,
                )
                loss = self.loss(loss_input_dict)
                opt_dict = self._optimize_step(loss)
                loss_time = time.time() - cur_time
                if self.log:
                    log_dict = dict(ChainMap(opt_dict, self.loss.loss_dict))
                    self._log(log_dict, epoch)
                pbar.update(1)
                pbar.desc = f"Epoch {epoch} | data_time: {data_time:.2f} | applyer_time: {applyer_time:.2f} | " \
                            f"model_time: {model_time:.2f} | loss_time: {loss_time:.2f}"
                cur_time = time.time()

    def test_video(self, video_dataset):
        """Test a single video."""
        results = []
        benign_results = []
        # render once for all frames
        # because we use fix renderer for evaluation
        render_patch = self.renderer(self.patch, train=False).detach().cpu()
        with torch.no_grad():
            for frame_idx, data in tqdm(enumerate(video_dataset), total=len(video_dataset), desc="Testing"):

                if frame_idx == 0:
                    # initialize the tracker
                    # [x1, y1, w, h]
                    print(f'Initialize tracker at frame {frame_idx} with bbox {data["init_bbox"]}')
                    self.tracker.initialize(data['img'], {'init_bbox': data['init_bbox']})
                else:
                    # track the target
                    if data['apply_attack']:
                        img = torch.tensor(data['img'], dtype=torch.float32).unsqueeze(0)
                        coords = torch.tensor(data['coords']).unsqueeze(0)
                        # test_mode=True to fix patch orientation
                        img, _ = self.applyer(img, render_patch, coords, test_mode=True)
                        img = img.squeeze(0).numpy()
                    else:
                        img = data['img']
                    if self.defense is not None:
                        img = self.defense(img)
                    out = self.tracker.track(img)

                # save the results
                if data['apply_attack']:
                    results.append(dict(
                        frame_idx=frame_idx,
                        out=out,
                        gt=dict(target_bbox=data['umbrella_bbox']),
                    ))
                # save benign results used for evaluation the 
                # benign performance gap with/without defense
                else:
                    if frame_idx == 0:
                        continue
                    benign_results.append(dict(
                        frame_idx=frame_idx,
                        out=out
                    ))
        return results, benign_results

    def test_epoch(self, epoch: int = -1):
        """Adversarial Example testing loop."""
        videos = getattr(self.test_loader, 'videos', None)
        results_dict = {}
        benign_results_dict = {}
        assert videos is not None, "Test loader must have a 'videos' attribute."
        for video_name, video_dataset in videos.items():
            print('Evaluating video:', video_name)
            results_dict[video_name], benign_results_dict[video_name] = self.test_video(video_dataset)
        # save results
        self._save_json(results_dict, epoch)
        if epoch == -1:
            self._save_json(benign_results_dict, epoch, prefix='benign_results')
        return results_dict

    def _optimize_step(self, loss):
        """Perform a single optimization step."""
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([self.patch], max_norm=float('inf'))
        patch_clone = self.patch.clone()
        self.optimizer.step()
        patch_diff = torch.norm(patch_clone - self.patch)

        # attention: first clip pixel then clip circle
        clip_pixel_inplace(self.patch)
        clip_circle_inplace(self.patch)
        self.patch.requires_grad_()

        output_dict = dict(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            patch_diff=patch_diff.item(),
            lr=self.optimizer.param_groups[0]['lr']
        )
        return output_dict

    def evaluate_epoch(self, results_dict, epoch: int = -1):
        """Attack Metric Evaluation."""
        metrics = self.eval_metric(results_dict)
        self._save_json(metrics, epoch, prefix='metrics')
        return metrics

    @staticmethod
    def evaluate(results_dict, sub_string: str, eval_metric: Callable):
        new_results_dict = {}
        for video_name, video_results in results_dict.items():
            if sub_string in video_name:
                new_results_dict[video_name] = video_results
        metric = eval_metric(new_results_dict)
        return metric
        

    def run_scheduler(self, epoch: int):
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def run(self, resume_epoch: int = 0):
        """Adversarial Example training loop."""
        for epoch in range(self.epochs):
            self.run_scheduler(epoch)
            if epoch < resume_epoch:
                continue
            self.train_epoch(epoch)
            self._save_patch(epoch)
            if (epoch + 1) % self.eval_interval == 0:
                results_dict = self.test_epoch(epoch)
                metric_dict = self.evaluate_epoch(results_dict, epoch)
                self._log(metric_dict, epoch)
            
    def _save_json(self, results_dict, epoch, prefix='results'):
        """Save the results to a file."""
        save_path = os.path.join(self.work_dir, 'json_files', f'{prefix}_epoch{epoch}.json')
        if os.path.exists(save_path):
            ans = input(f'File {save_path} already exists, overwrite it? (y/n).')
            if ans.lower() != 'y':
                save_path = save_path.replace('.json', '_new.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_dict, f)

    def _debug(self, out, data):
        """Debugging function."""
        debug_siamrpn = False
        if out[0]['pred_boxes'].squeeze().dim() == 3:
            debug_siamrpn = True
        mean = np.array(self.IMG_CONFIG['mean'])
        std = np.array(self.IMG_CONFIG['std'])
        template = data['template'][0].permute(1, 2, 0).cpu().numpy()
        template = (template * std + mean).astype(np.uint8)
        search = data['search'][0].permute(1, 2, 0).detach().cpu().numpy()
        search = (search * std + mean).astype(np.uint8)
        h, w = search.shape[:2]
        if debug_siamrpn:
            online_template = np.zeros_like(template)
        else:
            online_template = data['online_template'][0].permute(1, 2, 0).cpu().numpy()
            online_template = (online_template * std + mean).astype(np.uint8)
        gt_bbox = data['normalize_bbox'][0].cpu().numpy() * np.array([w, h, w, h])
        if debug_siamrpn:
            pred_bbox = out[0]['pred_boxes'][0].detach().cpu().numpy().squeeze()
            pred_score = out[0]['pred_scores'][0].detach().cpu().numpy().squeeze()
            max_score_idx = np.argmax(pred_score)
            pred_score = pred_score[max_score_idx]
            pred_bbox = pred_bbox[max_score_idx] * np.array([w, h, w, h])
            pred_bbox = np.array([pred_bbox[0] - pred_bbox[2] / 2, pred_bbox[1] - pred_bbox[3] / 2,
                                  pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2])
        else:
            pred_bbox = out[0]['pred_boxes'][0].detach().cpu().numpy().squeeze() * np.array([w, h, w, h])
            pred_bbox = np.array([pred_bbox[0] - pred_bbox[2] / 2, pred_bbox[1] - pred_bbox[3] / 2,
                                  pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2])
            pred_score = out[0]['pred_scores'][0].detach().cpu().numpy()
        gt_bbox = gt_bbox.astype(np.int32)
        pred_bbox = pred_bbox.astype(np.int32)
        search = cv2.rectangle(search.copy(), (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 1)
        search = cv2.rectangle(search, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 1)
        search = cv2.putText(search, f"{pred_score:.2f}", (pred_bbox[0], pred_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             (0, 0, 255), 2)
        cv2.imwrite('./debug/template.jpg', template)
        cv2.imwrite('./debug/search.jpg', search)
        cv2.imwrite('./debug/online_template.jpg', online_template)
        return

    def eval(self):
        results_dict = self.test_epoch(epoch=-1)
        metric_dict = self.evaluate_epoch(results_dict, epoch=-1)
        return metric_dict
