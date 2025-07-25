import torch
import yaml
from easydict import EasyDict as edict

from models.AlphaPose.alphapose.models.builder import build_sppe
from ..builder import MODEL

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

@MODEL.register_module(name='pose_estimator')
def get_pose_estimator(cfg, checkpoint):
    cfg = update_config(cfg)
    pose_estimator = build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (checkpoint,))
    pose_estimator.load_state_dict(torch.load(checkpoint, map_location='cuda'))
    pose_estimator = pose_estimator.to('cuda')
    pose_estimator.eval()
    return pose_estimator