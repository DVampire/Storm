import os
import torch
from typing import Any, Dict
from glob import glob
import json
import random
import numpy as np
from collections import Counter

from storm.registry import TRAINER
from storm.registry import DOWNSTREAM
from storm.utils import check_data
from storm.utils import SmoothedValue
from storm.utils import MetricLogger
from storm.metrics import MSE, RankICIR, RankIC
from storm.models import get_patch_info, patchify
from storm.utils import convert_int_to_timestamp
from storm.utils import save_joblib
from storm.utils import save_json
from storm.utils import Records

@TRAINER.register_module(force=True)
class DynamicDualVQVAETrainer():
    def __init__(self,
                 *args,
                 config,
                 vae,
                 vae_ema,
                 dit,
                 dit_ema,
                 diffusion,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 loss_funcs,
                 vae_optimizer,
                 dit_optimizer,
                 vae_scheduler,
                 dit_scheduler,
                 logger,
                 device,
                 dtype,
                 writer: Any = None,
                 wandb: Any = None,
                 plot: Any = None,
                 accelerator: Any = None,
                 print_freq: int = 20,
                 train_gather_multi_gpu = False,
                 **kwargs):

        self.config = config
        self.batch_size = self.config.batch_size
        self.start_epoch = self.config.start_epoch
        self.num_training_epochs = self.config.num_training_epochs
        self.num_valid_epochs = self.config.num_training_epochs
        self.num_testing_epochs = 1
        self.num_checkpoint_del = self.config.num_checkpoint_del
        self.checkpoint_period = self.config.checkpoint_period
        self.vae = vae
        self.vae_ema = vae_ema
        self.dit = dit
        self.dit_ema = dit_ema
        self.diffusion = diffusion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.vae_loss_fn = loss_funcs.get("vae_loss", None)
        self.price_cont_loss_fn = loss_funcs.get("price_cont_loss", None)
        self.vae_optimizer = vae_optimizer
        self.dit_optimizer = dit_optimizer
        self.vae_scheduler = vae_scheduler
        self.dit_scheduler = dit_scheduler
        self.logger = logger
        self.device = device
        self.dtype = dtype
        self.exp_path = config.exp_path
        self.writer = writer
        self.wandb = wandb
        self.num_plot_samples = self.config.num_plot_samples
        self.plot = plot
        self.accelerator = accelerator
        self.print_freq = print_freq
        self.train_gather_multi_gpu = train_gather_multi_gpu

        self.downstream = DOWNSTREAM.build(self.config.downstream)

        self._init_params()

    def _init_params(self):

        self.logger.info("| Init parameters for VAE trainer...")

        torch.set_default_dtype(self.dtype)

        self.is_main_process = self.accelerator.is_local_main_process

        self.model = self.accelerator.prepare(self.vae)
        self.model_ema = self.accelerator.prepare(self.vae_ema)

        self.model._set_static_graph()

        if self.vae_loss_fn:
            self.vae_loss_fn = self.accelerator.prepare(self.vae_loss_fn)
        if self.price_cont_loss_fn:
            self.price_cont_loss_fn = self.accelerator.prepare(self.price_cont_loss_fn)

        self.optimizer = self.accelerator.prepare(self.vae_optimizer)
        self.scheduler = self.accelerator.prepare(self.vae_scheduler)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        torch.set_default_dtype(self.dtype)

        self.checkpoint_path = os.path.join(self.exp_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.plot_path = os.path.join(self.exp_path, "plot")
        os.makedirs(self.plot_path, exist_ok=True)

        self.start_epoch = self.load_checkpoint() + 1

        self.check_batch_info_flag = True

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

    def save_checkpoint(self, epoch: int, if_best: bool = False):
        if not self.accelerator.is_local_main_process:
            return  # Only save checkpoint on the main process

        if if_best:
            checkpoint_file = os.path.join(self.checkpoint_path, f"best.pth")
        else:
            checkpoint_file = os.path.join(self.checkpoint_path, "checkpoint_{:06d}.pth".format(epoch))

        # Save model, optimizer, and scheduler states
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'model_ema_state': self.accelerator.unwrap_model(self.model_ema).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        self.accelerator.save(state, checkpoint_file)

        # Manage saved checkpoints
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(checkpoint_files) > self.num_checkpoint_del:
            for checkpoint_file in checkpoint_files[:-self.num_checkpoint_del]:
                os.remove(checkpoint_file)
                self.logger.info(f"｜ Checkpoint deleted: {checkpoint_file}")

        self.logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, epoch: int = -1, checkpoint_file: str = None, if_best: bool = False):

        if checkpoint_file is None:
            best_checkpoint_file = os.path.join(self.checkpoint_path, "best.pth")
            epoch_checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch:06d}.pth")

            latest_checkpoint_file = None
            checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
            if not checkpoint_files:
                self.logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
            else:
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint_file = checkpoint_files[-1]
                latest_checkpoint_file = checkpoint_file

            if if_best and os.path.exists(best_checkpoint_file):
                checkpoint_file = best_checkpoint_file
                self.logger.info(f"| Load best checkpoint: {checkpoint_file}")
            elif epoch >= 0 and os.path.exists(epoch_checkpoint_file):
                checkpoint_file = epoch_checkpoint_file
                self.logger.info(f"| Load epoch checkpoint: {checkpoint_file}")
            else:
                if latest_checkpoint_file:
                    checkpoint_file = latest_checkpoint_file
                    self.logger.info(f"| Load latest checkpoint: {checkpoint_file}")
                else:
                    checkpoint_file = None
                    self.logger.info(f"| Checkpoint not found.")
        else:
            self.logger.info(f"| Load checkpoint: {checkpoint_file}")

        if checkpoint_file is not None:

            state = torch.load(checkpoint_file, map_location=self.device)

            # Unwrap the model to load state dict into the underlying model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(state['model_state'])

            unwrapped_model_ema = self.accelerator.unwrap_model(self.model_ema)
            unwrapped_model_ema.load_state_dict(state['model_ema_state'])

            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])

            return state['epoch']
        else:
            return 0

    def __str__(self):
        return f"DynamicDualVQVAETrainer(num_training_epochs={self.num_training_epochs}, start_epoch={self.start_epoch}, batch_size={self.batch_size})"

    def check_batch_info(self, batch: Dict):

        if self.check_batch_info_flag:
            asset = batch["asset"]
            self.logger.info(f"| Asset: {check_data(asset)}")

            for key in batch:
                if key not in ["asset"]:
                    data = batch[key]
                    log_str = f"| {key}: "
                    for key, value in data.items():
                        log_str += f"\n {key}: {check_data(value)}"
                    self.logger.info(log_str)
            self.check_batch_info_flag = False

    def run_step(self,
                 epoch,
                 if_use_writer = True,
                 if_use_wandb = True,
                 if_plot = False,
                 mode = "train"
                 ):

        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        if_train = mode == "train"

        records = Records(accelerator=self.accelerator)
        metric_logger = MetricLogger(delimiter="  ")

        if if_train:
            self.model.train(True)
        else:
            self.model.eval()

        if self.accelerator.use_distributed:
            cs_patch_size = self.model.module.cs_patch_size
            cs_if_mask = self.model.module.cs_if_mask
            ts_patch_size = self.model.module.ts_patch_size
            ts_if_mask = self.model.module.ts_if_mask
        else:
            cs_patch_size = self.model.cs_patch_size
            cs_if_mask = self.model.cs_if_mask
            ts_patch_size = self.model.ts_patch_size
            ts_if_mask = self.model.ts_if_mask

        if mode == "train":
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

            header = f"| Train Epoch: [{epoch}/{self.num_training_epochs}]"
            global_step = self.global_train_step
            dataloader = self.train_dataloader
        elif mode == "valid":
            header = f"| Valid Epoch: [{epoch}/{self.num_valid_epochs}]"
            global_step = self.global_valid_step
            dataloader = self.valid_dataloader
        else:
            header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
            global_step = self.global_test_step
            dataloader = self.test_dataloader

        sample_batchs = []
        if if_plot:
            num_plot_sample_batch = int(self.num_plot_samples // self.plot.sample_num)
            sample_batchs = random.sample(range(len(dataloader)), num_plot_sample_batch)

        rankics = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1) # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0] # (N, C, T, S)

            # Forward
            if if_train:
                output = self.model(features, labels, training = True)
            else:
                with torch.no_grad():
                    output = self.model(features, labels, training = False)

            # Restore prices
            input_size = prices.shape

            cs_pred_prices = output["recon_cs"]
            cs_patch_size = (cs_patch_size[0], cs_patch_size[1], prices.shape[-1])
            cs_patch_info = get_patch_info(input_size, cs_patch_size)
            ts_pred_prices = output["recon_ts"]
            ts_patch_size = (ts_patch_size[0], ts_patch_size[1], prices.shape[-1])
            ts_patch_info = get_patch_info(input_size, ts_patch_size)

            restored_target_prices = prices * prices_std + prices_mean
            cs_restored_pred_prices = cs_pred_prices
            cs_restored_pred_prices = cs_restored_pred_prices * prices_std + prices_mean # (N, C, T, S, 5)
            ts_restored_pred_prices = ts_pred_prices
            ts_restored_pred_prices = ts_restored_pred_prices * prices_std + prices_mean # (N, C, T, S, 5)
            restored_pred_prices = self.vae_loss_fn.cs_scale * cs_restored_pred_prices + ts_restored_pred_prices

            cs_patched_target_prices = patchify(prices, patch_info=cs_patch_info)  # (N, L, D)
            cs_patched_pred_prices = patchify(cs_pred_prices, patch_info=cs_patch_info)  # (N, L, D)
            ts_patched_target_prices = patchify(prices, patch_info=ts_patch_info)  # (N, L, D)
            ts_patched_pred_prices = patchify(ts_pred_prices, patch_info=ts_patch_info)  # (N, L, D)

            # plot
            if if_plot and data_iter_step in sample_batchs:
                if self.is_main_process:
                    try:
                        save_dir = os.path.join(self.plot_path, "comparison_kline")
                        save_prefix = "{}_epoch_{:06d}_batch_{:06d}".format(mode, epoch, data_iter_step)
                        self.plot.plot_comparison_kline(
                            asset,
                            start_timestamp.detach().cpu().numpy(),
                            end_timestamp.detach().cpu().numpy(),
                            timestamps.detach().cpu().numpy(),
                            restored_target_prices.squeeze(1).detach().cpu().numpy(),
                            restored_pred_prices.squeeze(1).detach().cpu().numpy(),
                            save_dir=save_dir,
                            save_prefix=save_prefix
                        )
                    except Exception as e:
                        self.logger.error(f"Plot error: {e}")

            # Compute loss
            weighted_quantized_loss = output["weighted_quantized_loss"]
            weighted_clip_loss = output["weighted_clip_loss"]
            weighted_commit_loss = output["weighted_commit_loss"]
            weighted_codebook_diversity_loss = output["weighted_codebook_diversity_loss"]
            weighted_orthogonal_reg_loss = output["weighted_orthogonal_reg_loss"]

            loss += (weighted_quantized_loss +
                     weighted_clip_loss +
                     weighted_commit_loss +
                     weighted_codebook_diversity_loss +
                     weighted_orthogonal_reg_loss)

            records.update({
                "weighted_quantized_loss": weighted_quantized_loss,
                "weighted_clip_loss": weighted_clip_loss,
                "weighted_commit_loss": weighted_commit_loss,
                "weighted_codebook_diversity_loss": weighted_codebook_diversity_loss,
                "weighted_orthogonal_reg_loss": weighted_orthogonal_reg_loss
            })

            cs_mask = output["mask_cs"]
            ts_mask = output["mask_ts"]
            pred_label = output["pred_label"] # (N, S)
            posterior = output["posterior"]
            prior = output["prior"]
            labels = labels.squeeze(1).squeeze(1)  # (N, S)

            if self.vae_loss_fn:
                loss_dict = self.vae_loss_fn(cs_sample=cs_patched_pred_prices,
                                             cs_target_sample=cs_patched_target_prices,
                                             ts_sample=ts_patched_pred_prices,
                                             ts_target_sample=ts_patched_target_prices,
                                             pred_label=pred_label,
                                             label=labels,
                                             posterior=posterior,
                                             prior=prior,
                                             cs_mask=cs_mask,
                                             ts_mask=ts_mask,
                                             cs_if_mask=cs_if_mask,
                                             ts_if_mask=ts_if_mask)

                weighted_nll_loss = loss_dict["weighted_nll_loss"]
                weighted_cs_nll_loss = loss_dict["weighted_cs_nll_loss"]
                weighted_ts_nll_loss = loss_dict["weighted_ts_nll_loss"]
                weighted_kl_loss = loss_dict["weighted_kl_loss"]
                weighted_ret_loss = loss_dict["weighted_ret_loss"]

                records.update({
                    "weighted_nll_loss": weighted_nll_loss,
                    "weighted_cs_nll_loss": weighted_cs_nll_loss,
                    "weighted_ts_nll_loss": weighted_ts_nll_loss,
                    "weighted_kl_loss": weighted_kl_loss,
                    "weighted_ret_loss": weighted_ret_loss,
                })

                loss += weighted_nll_loss + weighted_kl_loss + weighted_ret_loss

            if self.price_cont_loss_fn:
                loss_dict = self.price_cont_loss_fn(prices=restored_pred_prices.squeeze(1))
                weighted_cont_loss = loss_dict["weighted_cont_loss"]

                records.update({
                    "weighted_cont_loss": weighted_cont_loss
                })

                loss += weighted_cont_loss

            records.update({
                "loss": loss
            })

            # compute metrics
            cs_restored_pred_prices = patchify(cs_restored_pred_prices, patch_info=cs_patch_info)
            cs_restored_target_prices = patchify(restored_target_prices, patch_info=cs_patch_info)
            ts_restored_pred_prices = patchify(ts_restored_pred_prices, patch_info=ts_patch_info)
            ts_restored_target_prices = patchify(restored_target_prices, patch_info=ts_patch_info)
            cs_restored_pred_prices = cs_restored_pred_prices.detach()
            cs_restored_target_prices = cs_restored_target_prices.detach()
            ts_restored_pred_prices = ts_restored_pred_prices.detach()
            ts_restored_target_prices = ts_restored_target_prices.detach()

            if cs_if_mask and ts_if_mask:
                cs_mask = cs_mask.repeat(1, 1, prices.shape[-1])
                cs_mask_target_prices = cs_restored_target_prices * cs_mask
                cs_mask_pred_prices = cs_restored_pred_prices * cs_mask
                cs_nomask_target_prices = cs_restored_target_prices * (1.0 - cs_mask)
                cs_nomask_pred_prices = cs_restored_pred_prices * (1.0 - cs_mask)

                cs_mask_mse = MSE(cs_mask_target_prices, cs_mask_pred_prices)
                cs_nomask_mse = MSE(cs_nomask_target_prices, cs_nomask_pred_prices)
                cs_mse = MSE(cs_restored_target_prices, cs_restored_pred_prices)

                ts_mask = ts_mask.repeat(1, 1, prices.shape[-1])
                ts_mask_target_prices = ts_restored_target_prices * ts_mask
                ts_mask_pred_prices = ts_restored_pred_prices * ts_mask
                ts_nomask_target_prices = ts_restored_target_prices * (1.0 - ts_mask)
                ts_nomask_pred_prices = ts_restored_pred_prices * (1.0 - ts_mask)

                ts_mask_mse = MSE(ts_mask_target_prices, ts_mask_pred_prices)
                ts_nomask_mse = MSE(ts_nomask_target_prices, ts_nomask_pred_prices)
                ts_mse = MSE(ts_restored_target_prices, ts_restored_pred_prices)

                mask_mse = self.vae_loss_fn.cs_scale * cs_mask_mse + ts_mask_mse
                nomask_mse = self.vae_loss_fn.cs_scale * cs_nomask_mse + ts_nomask_mse
                mse = self.vae_loss_fn.cs_scale * cs_mse + ts_mse

                records.update({
                    "mask_mse": mask_mse,
                    "nomask_mse": nomask_mse,
                    "mse": mse,
                    "cs_mask_mse": cs_mask_mse,
                    "cs_nomask_mse": cs_nomask_mse,
                    "cs_mse": cs_mse,
                    "ts_mask_mse": ts_mask_mse,
                    "ts_nomask_mse": ts_nomask_mse,
                    "ts_mse": ts_mse,
                })

            else:

                cs_mse = MSE(cs_restored_target_prices, cs_restored_pred_prices)
                ts_mse = MSE(ts_restored_target_prices, ts_restored_pred_prices)

                mse = self.vae_loss_fn.cs_scale * cs_mse + ts_mse

                records.update({
                    "mse": mse,
                    "cs_mse": cs_mse,
                    "ts_mse": ts_mse,
                })

            # pred_label = pred_label.detach()
            # labels = labels.detach()
            #
            # rankic = RankIC(pred_label, labels)
            # rankics.append(rankic)
            #
            # rankicir = RankICIR(rankics)
            #
            # records.update(data = {"RANKIC":rankic, "RANKICIR": rankicir },
            #                extra_info={"pred_label": pred_label,"true_label": labels})

            if if_train:
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]

                records.update_value({"lr": lr})

            # gather data from multi gpu
            records.gather(self.train_gather_multi_gpu)
            gathered_item = records.gathered_item

            global_step += 1

            prefix = mode
            if global_step % self.print_freq == 0:

                wandb_dict = {}
                # For records
                for key, value in gathered_item.items():
                    if if_use_writer and self.writer:
                        self.writer.log_scalar(f"{prefix}/{key}", value, global_step)
                    if if_use_wandb and self.wandb:
                        wandb_dict[f"{prefix}/{key}"] = value

                self.wandb.log(wandb_dict)

            metric_logger.update(**gathered_item)
            metric_logger.synchronize_between_processes()

        if if_use_writer and self.is_main_process:
            self.writer.flush()

        if mode == "train":
            self.global_train_step = global_step
            log_str = "| Train averaged stats: "
        elif mode == "valid":
            self.global_valid_step = global_step
            log_str = "| Valid averaged stats: "
        else:
            self.global_test_step = global_step
            log_str = "| Test averaged stats: "

        for name, meter in metric_logger.meters.items():
            log_str += f"- {name}: {meter}"
        self.logger.info(log_str)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def run_test(self,
                 epoch,
                 if_use_writer=True,
                 if_use_wandb=True,
                 if_plot=False,
                 mode="test"
                 ):

        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        records = Records(accelerator=self.accelerator)

        metric_logger = MetricLogger(delimiter="  ")
        self.model.eval()

        if self.accelerator.use_distributed:
            cs_patch_size = self.model.module.cs_patch_size
            ts_patch_size = self.model.module.ts_patch_size
        else:
            cs_patch_size = self.model.cs_patch_size
            ts_patch_size = self.model.ts_patch_size

        header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
        dataloader = self.test_dataloader

        sample_batchs = []
        if if_plot:
            num_plot_sample_batch = int(self.num_plot_samples // self.plot.sample_num)
            sample_batchs = random.sample(range(len(dataloader)), num_plot_sample_batch)

        rankics = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1)  # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0]  # (N, C, T, S)

            # Forward
            with torch.no_grad():
                output = self.model(features, labels, training=False)

            # Restore prices
            input_size = prices.shape

            cs_pred_prices = output["recon_cs"]
            cs_patch_size = (cs_patch_size[0], cs_patch_size[1], prices.shape[-1])
            cs_patch_info = get_patch_info(input_size, cs_patch_size)
            ts_pred_prices = output["recon_ts"]
            ts_patch_size = (ts_patch_size[0], ts_patch_size[1], prices.shape[-1])
            ts_patch_info = get_patch_info(input_size, ts_patch_size)

            restored_target_prices = prices * prices_std + prices_mean
            cs_restored_pred_prices = cs_pred_prices
            cs_restored_pred_prices = cs_restored_pred_prices * prices_std + prices_mean  # (N, C, T, S, 5)
            ts_restored_pred_prices = ts_pred_prices
            ts_restored_pred_prices = ts_restored_pred_prices * prices_std + prices_mean  # (N, C, T, S, 5)
            restored_pred_prices = self.vae_loss_fn.cs_scale * cs_restored_pred_prices + ts_restored_pred_prices

            # plot
            if if_plot and data_iter_step in sample_batchs:
                if self.is_main_process:
                    try:
                        save_dir = os.path.join(self.plot_path, "comparison_kline")
                        save_prefix = "{}_epoch_{:06d}_batch_{:06d}".format(mode, epoch, data_iter_step)
                        self.plot.plot_comparison_kline(
                            asset,
                            start_timestamp.detach().cpu().numpy(),
                            end_timestamp.detach().cpu().numpy(),
                            timestamps.detach().cpu().numpy(),
                            restored_target_prices.squeeze(1).detach().cpu().numpy(),
                            restored_pred_prices.squeeze(1).detach().cpu().numpy(),
                            save_dir=save_dir,
                            save_prefix=save_prefix
                        )
                    except Exception as e:
                        self.logger.error(f"Plot error: {e}")

            pred_label = output["pred_label"]  # (N, S)
            labels = labels.squeeze(1).squeeze(1)  # (N, S)


            # Compute metrics
            cs_restored_pred_prices = patchify(cs_restored_pred_prices, patch_info=cs_patch_info)
            cs_restored_target_prices = patchify(restored_target_prices, patch_info=cs_patch_info)
            ts_restored_pred_prices = patchify(ts_restored_pred_prices, patch_info=ts_patch_info)
            ts_restored_target_prices = patchify(restored_target_prices, patch_info=ts_patch_info)
            cs_restored_pred_prices = cs_restored_pred_prices.detach()
            cs_restored_target_prices = cs_restored_target_prices.detach()
            ts_restored_pred_prices = ts_restored_pred_prices.detach()
            ts_restored_target_prices = ts_restored_target_prices.detach()

            cs_mse = MSE(cs_restored_target_prices, cs_restored_pred_prices)
            ts_mse = MSE(ts_restored_target_prices, ts_restored_pred_prices)
            mse = self.vae_loss_fn.cs_scale * cs_mse + ts_mse

            records.update({
                "MSE": mse,
                "CS_MSE": cs_mse,
                "TS_MSE": ts_mse,
            })

            pred_label = pred_label.detach()
            labels = labels.detach()
            end_timestamp = end_timestamp.detach()

            rankic = RankIC(pred_label, labels)
            rankics.append(rankic)

            rankicir = RankICIR(rankics)

            records.update(data={"RANKIC": rankic, "RANKICIR": rankicir},
                           extra_info={"end_timestamp": end_timestamp,
                                       "pred_label": pred_label, "true_label": labels})

            # gather data from multi gpu
            records.gather()

        gathered_item = records.gathered_item
        combiner = records.combiner
        extra_combiner = records.extra_combiner

        # Process records
        metrics = dict()
        for key, values in combiner.items():
            metrics[key] = np.mean(values)
        metrics["RANKICIR"] = gathered_item["RANKICIR"]

        # Process extra records
        if self.is_main_process:
            end_timestamps = extra_combiner["end_timestamp"]
            pred_labels = extra_combiner["pred_label"]
            true_labels = extra_combiner["true_label"]

            end_timestamps = np.concatenate(end_timestamps, axis=0)
            pred_labels = np.concatenate(pred_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)

            # sort according to end_timestamp
            indices = np.argsort(end_timestamps)
            end_timestamps = end_timestamps[indices]
            pred_labels = pred_labels[indices]
            true_labels = true_labels[indices]

            downstream_metrics = self.downstream(pred_labels = pred_labels,
                                                 true_labels = true_labels)

            metrics.update({
                "CW": downstream_metrics["CW"],
                "ARR%": downstream_metrics["ARR%"],
                "SR": downstream_metrics["SR"],
                "CR": downstream_metrics["CR"],
                "SOR": downstream_metrics["SOR"],
                "DD": downstream_metrics["DD"],
                "MDD%": downstream_metrics["MDD%"],
                "VOL": downstream_metrics["VOL"],
            })

        # Round
        for key, value in metrics.items():
            metrics[key] = float(np.round(value, 4))

        prefix = mode
        wandb_dict = {}
        # For records
        for key, value in metrics.items():
            if if_use_writer and self.writer:
                self.writer.log_scalar(f"{prefix}/{key}", value, epoch)
            if if_use_wandb and self.wandb:
                wandb_dict[f"{prefix}/{key}"] = value
        self.wandb.log(wandb_dict)

        if if_use_writer and self.is_main_process:
            self.writer.flush()

        log_str = "| Test averaged stats: "

        for name, meter in metrics.items():
            log_str += f"- {name}: {meter}"
        self.logger.info(log_str)

        return metrics

    def run_state(self,
                 epoch,
                 mode="test"
                 ):

        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        records = Records(accelerator=self.accelerator)

        metric_logger = MetricLogger(delimiter="  ")
        self.model.eval()

        if self.accelerator.use_distributed:
            cs_n_size = self.model.module.cs_embed_layer.n_size
            ts_n_size = self.model.module.ts_embed_layer.n_size
            cs_n_num = self.model.module.cs_embed_layer.n_num
            ts_n_num = self.model.module.ts_embed_layer.n_num
        else:
            cs_n_size = self.model.embed_layer.n_size
            ts_n_size = self.model.embed_layer.n_size
            cs_n_num = self.model.embed_layer.n_num
            ts_n_num = self.model.embed_layer.n_num

        header = f"| State Epoch: [{epoch}/{self.num_testing_epochs}]"
        dataloader = self.test_dataloader

        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1)  # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0]  # (N, C, T, S)

            # Forward
            with torch.no_grad():
                output = self.model(features, labels, training=False)

            factors_cs = output["factors_cs"]
            factors_ts = output["factors_ts"]
            embed_ind_cs = output["embed_ind_cs"]
            embed_ind_ts = output["embed_ind_ts"]

            factors_cs = factors_cs.detach()
            factors_ts = factors_ts.detach()
            embed_ind_cs = embed_ind_cs.detach()
            embed_ind_ts = embed_ind_ts.detach()

            end_timestamp = end_timestamp.detach()

            records.update(extra_info={"end_timestamps": end_timestamp,
                                       "factors_cs": factors_cs,
                                       "factors_ts": factors_ts,
                                       "embed_ind_cs": embed_ind_cs,
                                       "embed_ind_ts": embed_ind_ts})

            # gather data from multi gpu
            records.gather(train_gather_multi_gpu = True)

        extra_combiner = records.extra_combiner

        # Process extra records
        if self.is_main_process:

            end_timestamps = extra_combiner["end_timestamps"]
            factors_cs = extra_combiner["factors_cs"]
            factors_ts = extra_combiner["factors_ts"]
            embed_ind_cs = extra_combiner["embed_ind_cs"]
            embed_ind_ts = extra_combiner["embed_ind_ts"]

            end_timestamps = np.concatenate(end_timestamps, axis=0)
            factors_cs = np.concatenate(factors_cs, axis=0)
            factors_ts = np.concatenate(factors_ts, axis=0)
            embed_ind_cs = np.concatenate(embed_ind_cs, axis=0)
            embed_ind_ts = np.concatenate(embed_ind_ts, axis=0)

            # count embedding index
            embed_ind_cs = embed_ind_cs.flatten().tolist()
            embed_ind_ts = embed_ind_ts.flatten().tolist()
            count_embed_ind_cs = Counter(embed_ind_cs)
            count_embed_ind_ts = Counter(embed_ind_ts)

            # sort according to end_timestamp
            indices = np.argsort(end_timestamps)
            end_timestamps = end_timestamps[indices]
            factors_cs = factors_cs[indices]
            factors_ts = factors_ts[indices]

            end_timestamps = end_timestamps.tolist()
            end_timestamps = [convert_int_to_timestamp(end_timestamps).strftime("%Y-%m-%d") for end_timestamps in end_timestamps]

            meta = {
                "cs_n_size": cs_n_size,
                "ts_n_size": ts_n_size,
                "cs_n_num": cs_n_num,
                "ts_n_num": ts_n_num,
                "factors_cs_shape": factors_cs.shape,
                "factors_ts_shape": factors_ts.shape,
                "timestamps": end_timestamps,
                "nums": len(end_timestamps),
            }

            print(meta)

            items = {}
            for end_timestamp, factor_cs, factor_ts in zip(end_timestamps, factors_cs, factors_ts):
                items[end_timestamp] = {
                    "factor_cs": factor_cs,
                    "factor_ts": factor_ts
                }

            count = {
                "count_cs": count_embed_ind_cs,
                "count_ts": count_embed_ind_ts
            }

            info = {
                "meta": meta,
                "items": items,
                "count": count
            }

            self.logger.info(f"| State info")
            if self.is_main_process:
                save_joblib(info, os.path.join(self.exp_path, f"state.joblib"))
                save_json(count, os.path.join(self.exp_path, f"count.json"))

    def train(self):

        self.logger.info("| Start training and evaluating VAE...")

        min_metric = float("inf")

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):
            train_stats = self.run_step(epoch, mode="train")
            valid_stats = self.run_step(epoch, mode="valid")
            test_stats = self.run_test(epoch, mode="test")

            metric = valid_stats["mse"]

            log_stats = {"epoch": epoch}
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_log.txt"), "a",) as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)

            if metric < min_metric:
                min_metric = metric
                self.save_checkpoint(epoch, if_best=True)

    def test(self, checkpoint_path: str = None):
        self.logger.info("| Start testing VAE...")

        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
            if os.path.exists(best_checkpoint_path):
                epoch = self.load_checkpoint(if_best=True)
            else:
                self.logger.info("| Best checkpoint not found. Load the last checkpoint.")
                epoch = self.load_checkpoint()
        else:
            epoch = self.load_checkpoint(checkpoint_file=checkpoint_path)

        log_stats = {"epoch": epoch}

        train_stats = self.run_step(epoch,
                                   mode="train",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        valid_stats = self.run_step(epoch,
                                   mode="valid",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        test_stats = self.run_test(epoch,
                                   mode="test",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
        log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "test_log.txt"), "w",) as f:
                f.write(json.dumps(log_stats) + "\n")

        self.logger.info("| Test finished.")

    def state(self, checkpoint_path: str = None):
        self.logger.info("| Start state VAE...")

        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
            if os.path.exists(best_checkpoint_path):
                epoch = self.load_checkpoint(if_best=True)
            else:
                self.logger.info("| Best checkpoint not found. Load the last checkpoint.")
                epoch = self.load_checkpoint()
        else:
            epoch = self.load_checkpoint(checkpoint_file=checkpoint_path)

        log_stats = {"epoch": epoch}

        self.run_state(epoch)

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "state_log.txt"), "w",) as f:
                f.write(json.dumps(log_stats) + "\n")

        self.logger.info("| State finished.")