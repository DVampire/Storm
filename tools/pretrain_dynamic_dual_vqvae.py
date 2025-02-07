import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from mmengine import DictAction
from copy import deepcopy
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
import pathlib
from dotenv import load_dotenv

load_dotenv(verbose=True)

root = str(pathlib.Path(__file__).resolve().parents[1])
current = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(current)
sys.path.append(root)

from storm.config import build_config
from storm.log import logger
from storm.log import tensorboard_logger
from storm.log import wandb_logger
from storm.registry import DATASET
from storm.registry import COLLATE_FN
from storm.registry import TRAINER
from storm.registry import MODEL
from storm.registry import DIFFUSION
from storm.registry import OPTIMIZER
from storm.registry import SCHEDULER
from storm.registry import LOSS_FUNC
from storm.registry import PLOT
from storm.data import prepare_dataloader
from storm.utils import assemble_project_path
from storm.utils import to_torch_dtype
from storm.utils import get_model_numel
from storm.utils import requires_grad
from storm.utils import record_model_param_shape

def get_args_parser():
    parser = argparse.ArgumentParser(description="Train script for storm")
    parser.add_argument("--config", default=os.path.join("configs", "exp", "pretrain_day_dj30_dynamic_dual_vqvae.py"), help="pretrain config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--tensorboard_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=False)

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.", )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--no_train", action="store_false", dest="train")
    parser.set_defaults(train=True)

    parser.add_argument("--test", action="store_true", help="test model")
    parser.add_argument("--no_test", action="store_false", dest="test")
    parser.set_defaults(test=True)

    parser.add_argument("--state", action="store_true", help="state model")
    parser.add_argument("--no_state", action="store_false", dest="state")
    parser.set_defaults(state=True)

    parser.add_argument("--tensorboard", action="store_true", default=True, help="enable tensorboard")
    parser.add_argument("--no_tensorboard", action="store_false", dest="tensorboard")
    parser.set_defaults(writer=True)

    parser.add_argument("--wandb", action="store_true", default=True, help="enable wandb")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb")
    parser.set_defaults(wandb=True)

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no_distributed", action="store_false", dest="distributed")
    parser.set_defaults(distributed=False)

    return parser

def main(args):

    # 1. build config
    config = build_config(assemble_project_path(args.config), args)

    # 2. set dtype
    dtype = to_torch_dtype(config.dtype)

    # 3. init accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=True if args.device == "cpu" else False, kwargs_handlers=[ddp_kwargs])

    # 4. get device
    device = accelerator.device

    # 5. init logger
    logger.init_logger(config.log_file, accelerator=accelerator)
    if config.tensorboard:
        tensorboard_logger.init_logger(config.tensorboard_path, accelerator=accelerator)
    if config.wandb:
        wandb_logger.init_logger(
            project=config.project,
            name=config.tag,
            config=config.to_dict(),
            dir=config.wandb_path,
            accelerator=accelerator,
        )

    collate_fn = COLLATE_FN.build(config.collate_fn)

    # 7. build dataset and dataloader
    train_dataset = DATASET.build(config.train_dataset)
    logger.info(f"| Train dataset: \n{train_dataset}")
    train_dataloader_args = dict(
        accelerator=accelerator,
        collate_fn=collate_fn,
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=config.pin_mem,
        distributed=config.distributed,
        train=True,
    )
    train_dataloader = prepare_dataloader(**train_dataloader_args)

    # update the config with the actual dataset size
    if args.device != "cpu" and torch.cuda.is_available():
        num_device = torch.cuda.device_count()
    else:
        num_device = 1

    num_training_data = len(train_dataset)
    num_training_steps_per_epoch = int(np.floor(num_training_data / (config.batch_size * num_device)))
    num_training_steps = int(num_training_steps_per_epoch * config.num_training_epochs)
    num_training_warmup_steps = int(num_training_steps_per_epoch * config.num_training_warmup_epochs)
    config.merge_from_dict({
        "num_training_data": num_training_data,
        "num_training_steps_per_epoch": num_training_steps_per_epoch,
        "num_training_steps": num_training_steps,
        "num_training_warmup_steps": num_training_warmup_steps,
    })

    valid_dataset = DATASET.build(config.valid_dataset)
    logger.info(f"| Valid dataset: \n{valid_dataset}")
    valid_dataloader_args = dict(
        accelerator=accelerator,
        collate_fn=collate_fn,
        dataset=valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=config.pin_mem,
        distributed=config.distributed,
        train=False,
    )
    valid_dataloader = prepare_dataloader(**valid_dataloader_args)

    test_dataset = DATASET.build(config.test_dataset)
    logger.info(f"| Test dataset: \n{test_dataset}")
    test_dataloader_args = dict(
        accelerator=accelerator,
        collate_fn=collate_fn,
        dataset=test_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=config.pin_mem,
        distributed=config.distributed,
        train=False,
    )
    test_dataloader = prepare_dataloader(**test_dataloader_args)

    # 8. build models
    vae = MODEL.build(config.vae)
    logger.info(f"| VAE: \n{vae}")
    vae_model_numel, vae_model_numel_trainable = get_model_numel(vae)
    logger.info(f"| VAE model numel: {vae_model_numel}, trainable: {vae_model_numel_trainable}")

    dit = MODEL.build(config.dit)
    logger.info(f"| DIT: \n{dit}")
    dit_model_numel, dit_model_numel_trainable = get_model_numel(dit)
    logger.info(f"| DIT model numel: {dit_model_numel}, trainable: {dit_model_numel_trainable}")

    diffusion = DIFFUSION.build(config.diffusion)
    logger.info(f"| Diffusion: \n{diffusion}")

    # 9. build ema
    vae_state_dict = vae.state_dict()
    vae_ema = MODEL.build(config.vae)
    vae_ema.load_state_dict(vae_state_dict)
    vae_ema = vae_ema.to(device, dtype)
    requires_grad(vae_ema, True)
    vae_ema_shape_dict = record_model_param_shape(vae_ema)
    logger.info(f"| VAE EMA: \n{vae_ema}")
    logger.info("| VAE EMA shape: \n{}".format("\n".join([f"{k}: {v}" for k, v in vae_ema_shape_dict.items()])))

    dit_state_dict = dit.state_dict()
    dit_ema = MODEL.build(config.dit)
    dit_ema.load_state_dict(dit_state_dict)
    dit_ema = dit_ema.to(device, dtype)
    requires_grad(dit_ema, True)
    dit_ema_shape_dict = record_model_param_shape(dit_ema)
    logger.info(f"| DIT EMA: \n{dit_ema}")
    logger.info("| DIT EMA shape: \n{}".format("\n".join([f"{k}: {v}" for k, v in dit_ema_shape_dict.items()])))

    # 10. move to device
    vae = vae.to(device, dtype)
    dit = dit.to(device, dtype)

    # 11. build loss function

    loss_funcs_config = config.loss_funcs_config
    loss_funcs = {}
    for loss_func_name, loss_func_config in loss_funcs_config.items():
        loss_funcs[loss_func_name] = LOSS_FUNC.build(loss_func_config).to(device, dtype)
        logger.info(f"| {loss_func_name} loss function: \n{loss_funcs[loss_func_name]}")

    # 12. build optimizer
    vae_params_groups = vae.parameters()
    vae_params_groups = filter(lambda p: p.requires_grad, vae_params_groups)
    vae_optimizer_config = deepcopy(config.vae_optimizer)
    vae_optimizer_config["params"] = vae_params_groups
    vae_optimizer = OPTIMIZER.build(vae_optimizer_config)
    logger.info(f"| VAE optimizer: \n{vae_optimizer}")

    dit_params_groups = dit.parameters()
    dit_params_groups = filter(lambda p: p.requires_grad, dit_params_groups)
    dit_optimizer_config = deepcopy(config.dit_optimizer)
    dit_optimizer_config["params"] = dit_params_groups
    dit_optimizer = OPTIMIZER.build(dit_optimizer_config)
    logger.info(f"| DIT optimizer: \n{dit_optimizer}")

    # 13. build lr scheduler
    vae_scheduler_config = deepcopy(config.vae_scheduler)
    vae_scheduler_config.update({
        "num_training_steps": config.num_training_steps,
        "num_warmup_steps": config.num_training_warmup_steps,
        "optimizer": vae_optimizer
    })
    vae_scheduler = SCHEDULER.build(vae_scheduler_config)
    logger.info(f"| VAE scheduler: \n{vae_scheduler}")

    dit_scheduler_config = deepcopy(config.dit_scheduler)
    dit_scheduler_config.update({
        "num_training_steps": config.num_training_steps,
        "num_warmup_steps": config.num_training_warmup_steps,
        "optimizer": dit_optimizer
    })
    dit_scheduler = SCHEDULER.build(dit_scheduler_config)
    logger.info(f"| DIT scheduler: \n{dit_scheduler}")

    # 14. build plot
    if hasattr(config, "plot"):
        plot = PLOT.build(config.plot)
    else:
        plot = None

    # 15. build trainer
    trainer_config = deepcopy(config.trainer)
    trainer_config.update({
        "config": config,
        "vae": vae,
        "vae_ema": vae_ema,
        "dit": dit,
        "dit_ema": dit_ema,
        "diffusion": diffusion,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "loss_funcs": loss_funcs,
        "vae_optimizer": vae_optimizer,
        "dit_optimizer": dit_optimizer,
        "vae_scheduler": vae_scheduler,
        "dit_scheduler": dit_scheduler,
        "logger": logger,
        "device": device,
        "dtype": dtype,
        "writer": tensorboard_logger,
        "wandb": wandb_logger,
        "plot": plot,
        "accelerator": accelerator,
    })
    trainer = TRAINER.build(trainer_config)
    logger.info(f"| Trainer: \n{trainer}")

    # 16. start training + evaluation
    logger.info(f"| Train: {args.train}")
    if args.train:
        trainer.train()

    # 17. start testing
    logger.info(f"| Test: {args.test}")
    if args.test:
        trainer.test(checkpoint_path = args.checkpoint_path)

    # 18. start state
    logger.info(f"| State: {trainer.state}")
    if args.state:
        trainer.state(checkpoint_path = args.checkpoint_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)