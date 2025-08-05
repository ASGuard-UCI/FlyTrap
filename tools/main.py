import argparse
import os
import shutil

import torch.cuda
from mmengine import Config
from mmengine.dataset import Compose
from mmengine.registry import DATASETS
from mmengine.runner import Runner

import flytrap.builder as builder
import flytrap.utils as utils
from flytrap.runner import AdversarialPatchRunner

utils.init_seeds(0)


def parse_cfg_options(cfg_options):
    """Parse config options from command line.
    
    Args:
        cfg_options (list): List of config options in format ['key=value', ...]
        
    Returns:
        dict: Parsed config dictionary
    """
    cfg_dict = {}
    for option in cfg_options:
        if '=' not in option:
            raise ValueError(f"Invalid config option format: {option}. Expected 'key=value'")
        
        key, value = option.split('=', 1)
        
        # Try to parse value as different types
        # First try boolean
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        # Try integer
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            value = int(value)
        # Try float
        else:
            try:
                value = float(value)
            except ValueError:
                # Keep as string if can't parse as number
                pass
        
        # Handle nested keys with dot notation
        keys = key.split('.')
        current = cfg_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return cfg_dict


def merge_cfg_dict(cfg, cfg_dict):
    """Recursively merge config dictionary into MMEngine Config object.
    
    Args:
        cfg: MMEngine Config object
        cfg_dict: Dictionary with config overrides
    """
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            # If the attribute doesn't exist, create it
            if not hasattr(cfg, key):
                setattr(cfg, key, Config({}))
            merge_cfg_dict(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)


def main(config, cfg_options=None):
    cfg = Config.fromfile(config)
    
    # Merge command line config options if provided
    if cfg_options:
        cfg_dict = parse_cfg_options(cfg_options)
        merge_cfg_dict(cfg, cfg_dict)

    # work_dir = cfg.work_dir
    work_dir = 'work_dirs/' + os.path.basename(config).split('.')[0]
    cfg.log_cfg['name'] = os.path.basename(config).split('.')[0]
    patch = utils.init_patch(cfg.patch_size, cfg.img_config)
    # load target patch as initialization during training
    if not cfg.eval:
        if os.path.exists(cfg.patch_path):
            patch = utils.load_patch(cfg.patch_path)
            patch_path = os.path.join(work_dir, 'patch.png')
        else:
            patch_path = os.path.join(work_dir, os.path.basename(cfg.patch_path))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    else:
        if not cfg.eval and not cfg.debug:
            ans = input(f"[Warning] Work_dir {work_dir} already exists, do you want to overwrite it? (y/n)")
            if ans.lower() != 'y':
                exit(0)
    # only load when evaluation
    # otherwise, the patch will be initialized
    if (cfg.eval or cfg.debug):
        # load patch from work_dir
        patch_path = cfg.patch_path
        if os.path.basename(patch_path) == patch_path:
            patch_path = os.path.join(work_dir, patch_path)
        assert os.path.exists(patch_path), f"Patch path {patch_path} not exists for evaluation or debug!"
        patch = utils.load_patch(patch_path)
        patch_file = os.path.basename(patch_path)
        if not os.path.exists(os.path.join(work_dir, patch_file)):
            shutil.copy(patch_path, os.path.join(work_dir, patch_file))
            patch_path = os.path.join(work_dir, patch_file)
    save_path = os.path.join(work_dir, os.path.basename(config))
    print('Dumping config to work_dir: %s' % save_path)
    cfg.dump(save_path)

    model = builder.MODEL.build(cfg.model)
    tracker = builder.MODEL.build(cfg.tracker)
    # tracker = None
    applyer = builder.APPLYER.build(cfg.applyer)
    renderer = builder.APPLYER.build(cfg.renderer)
    # transform of patch to be physical robust
    patch_transform = Compose(cfg.patch_transform)
    post_transform = Compose(cfg.post_transform)
    ## build optimizer and scheduler inside runner
    # optimizer = builder.OPTIMIZER.build(cfg.optimizer, patch)
    # scheduler = builder.SCHEDULER.build(cfg.scheduler, optimizer)
    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    test_loader = DATASETS.build(cfg.test_dataset)
    loss = builder.LossWrapper(cfg.loss)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = cfg.epochs
    metric = builder.METRICS.build(cfg.eval_metric)

    # debug
    # a = train_loader.dataset[5]

    runner = AdversarialPatchRunner(
        model=model,
        tracker=tracker,
        applyer=applyer,
        renderer=renderer,
        work_dir=work_dir,
        config=cfg,
        device=device,
        epochs=epochs,
        patch_path=patch_path,
        patch=patch,
        patch_transform=patch_transform,
        post_transform=post_transform,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_interval=cfg.eval_interval,
        eval_metric=metric,
        loss=loss,
        log=cfg.log,
        log_cfg=cfg.log_cfg,
        debug=cfg.debug,
        defense=getattr(cfg, 'defense', None)
    )

    if cfg.eval:
        print(runner.eval())
        exit()
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--cfg-options', nargs='*', default=[], 
                       help='Override config options using key=value format. '
                            'Examples: --cfg-options eval=true epochs=100 train_dataloader.batch_size=16')
    
    args = parser.parse_args()
    main(args.config, args.cfg_options)
