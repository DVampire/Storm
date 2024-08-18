import os
from mmengine import Config
from argparse import Namespace

from storm.utils import init_before_training
from storm.utils import assemble_project_path
from storm.log import warning

def build_config(config_path: str, args: Namespace) -> Config:
    config = Config.fromfile(filename=config_path)

    if args.cfg_options is None:
        cfg_options = dict()
    else:
        cfg_options = dict()
        for cfg_option in args.cfg_options:
            key, value = cfg_option.split("=")
            cfg_options[key] = value
    for item in args.__dict__:
        if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
            cfg_options[item] = args.__dict__[item]

    config.merge_from_dict(cfg_options)

    config.exp_path = assemble_project_path(os.path.join(config.workdir, config.tag))
    if config.if_remove is None:
        config.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {config.exp_path}? ") == 'y')
    if config.if_remove:
        import shutil
        shutil.rmtree(config.exp_path, ignore_errors=True)
        warning(f"| Arguments Remove work_dir: {config.exp_path}")
    else:
        warning(f"| Arguments Keep work_dir: {config.exp_path}")
    os.makedirs(config.exp_path, exist_ok=True)

    config.log_path = os.path.join(config.exp_path, config.log_file)
    warning(f"| Arguments Log file: {config.log_path}")

    config.tensorboard_path = os.path.join(config.exp_path, config.tensorboard_path)
    os.makedirs(config.tensorboard_path, exist_ok=True)
    warning(f"| Arguments Tensorboard path: {config.tensorboard_path}")

    config.checkpoint_path = os.path.join(config.exp_path, config.checkpoint_path)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    warning(f"| Arguments Checkpoint path: {config.checkpoint_path}")

    config.wandb_path = os.path.join(config.exp_path, config.wandb_path)
    os.makedirs(config.wandb_path, exist_ok=True)
    warning(f"| Arguments Wandb path: {config.wandb_path}")

    init_before_training(config.seed)
    warning(f"| Arguments Seed: {config.seed}")
    
    return config