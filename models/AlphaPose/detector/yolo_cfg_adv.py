from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'models/AlphaPose/detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'models/AlphaPose/detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM =  608
cfg.NMS_THRES =  0.99
cfg.CONFIDENCE = 0.01
cfg.NUM_CLASSES = 80
