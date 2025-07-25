import argparse
import json
import os
import cv2

from mmengine import Config
from tqdm import tqdm

import flytrap.builder as builder
from flytrap.runner import AdversarialPatchRunner


def main(args):
    config = args.config
    result_json = args.result_json
    sub_string = args.sub_string
    cfg = Config.fromfile(config)
    metric_fn = builder.METRICS.build(cfg.eval_metric)
    with open(result_json, 'r') as f:
        results_dict = json.load(f)
    print(AdversarialPatchRunner.evaluate(results_dict, sub_string, metric_fn))
    


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('config', type=str, help='metric config file')
    argparse.add_argument('result_json', type=str, help='Generated result json file to be evaluated')
    argparse.add_argument('--sub_string', type=str, default='', help='Sub string match to filter the results json file')
    args = argparse.parse_args()
    main(args)