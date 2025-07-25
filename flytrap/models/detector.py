from models.AlphaPose.detector.yolo_api import YOLODetector


import argparse
from ..builder import MODEL

parser = argparse.ArgumentParser(description='AlphaPose Demo')


@MODEL.register_module(name='yolo')
def get_yolo_detector(args):
    # args.gpus = [0]
    # args.device = 'cuda'
    from models.AlphaPose.detector.yolo_cfg import cfg
    detector = YOLODetector(cfg, args)
    return detector


@MODEL.register_module(name='yolo_adv')
def get_yolo_detector(args):
    # args.gpus = [0]
    # args.device = 'cuda'
    from models.AlphaPose.detector.yolo_cfg_adv import cfg
    detector = YOLODetector(cfg, args)
    return detector