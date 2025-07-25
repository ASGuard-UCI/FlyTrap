import importlib

import torch
import torch.nn as nn

from lib.models.mixformer_cvt import build_mixformer_cvt_online_score, build_mixformer_cvt_online_score_randomhead
from lib.models.mixformer_convmae import build_mixformer_convmae_online_score
import models.MixFormer.lib.test.tracker.mixformer_cvt_online as mixformer_cvt_online
import models.MixFormer.lib.test.tracker.mixformer_convmae_online as mixformer_convmae_online
import models.MixFormer.lib.utils.box_ops as box_ops
from ..builder import MODEL


def get_parameters(name, parameter_name, tracker_params=None):
    """Get parameters. Used for building tracker"""
    param_module = importlib.import_module('models.MixFormer.lib.test.parameter.{}'.format(name))
    search_area_scale = None
    if tracker_params is not None and 'search_area_scale' in tracker_params:
        search_area_scale = tracker_params['search_area_scale']
    model = ''
    if tracker_params is not None and 'model' in tracker_params:
        model = tracker_params['model']
    params = param_module.parameters(parameter_name, model, search_area_scale)
    if tracker_params is not None:
        for param_k, v in tracker_params.items():
            setattr(params, param_k, v)
    return params


def custom_get_parameters(name, parameter_name, model, search_area_scale):
    """Get parameters. Used for building model in training."""
    param_module = importlib.import_module('models.MixFormer.lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name, model, search_area_scale)
    return param_module


@MODEL.register_module(name='mixformer_cvt_online_scores')
def get_mixformer_cvt_online_scores(checkpoint):
    param_module = custom_get_parameters('mixformer_cvt_online', 'baseline', 'mixformer_online_22k.pth.tar', 4.5)
    cfg = param_module.cfg
    network = build_mixformer_cvt_online_score(cfg, train=False)
    mesg = network.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=False)
    print(mesg)
    return network


@MODEL.register_module(name='mixformer_cvt_online_scores_random_score_head')
def get_mixformer_cvt_online_scores_random_score_head(checkpoint):
    param_module = custom_get_parameters('mixformer_cvt_online', 'baseline', 'mixformer_online_22k.pth.tar', 4.5)
    cfg = param_module.cfg
    network = build_mixformer_cvt_online_score_randomhead(cfg, train=False)
    mesg = network.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=False)
    print(mesg)
    return network


@MODEL.register_module(name='mixformer_cvt_online_scores_tracker')
def get_mixformer_cvt_online_scores_tracker(params):
    params = get_parameters('mixformer_cvt_online', 'baseline', params)
    params.debug = True
    tracker = mixformer_cvt_online.MixFormerOnline(params, 'video')
    return tracker


@MODEL.register_module(name='mixformer_covmae_base_online_scores')
def get_mixformer_covmae_base_online_scores(checkpoint):
    param_module = custom_get_parameters('mixformer_convmae_online', 'baseline', 'mixformer_online_22k.pth.tar', 4.5)
    cfg = param_module.cfg
    network = build_mixformer_convmae_online_score(cfg, train=False)
    mesg = network.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=False)
    print(mesg)
    return network


@MODEL.register_module(name='mixformer_covmae_base_online_scores_tracker')
def get_mixformer_covmae_base_online_scores_tracker(params):
    params = get_parameters('mixformer_convmae_online', 'baseline', params)
    params.debug = True
    tracker = mixformer_convmae_online.MixFormerOnline(params, 'UAV')
    return tracker
