import glob
import builtins
import os
import psutil
import os.path as osp
import warnings
from typing import Union, Tuple
import torch
import torch.distributed as dist
import random
import numpy as np
from torch import inf
from mmengine.config import Config, ConfigDict
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
from dateutil.relativedelta import relativedelta
import math

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0

def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape

def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

def init_before_training(seed = 3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

def generate_intervals(start_date, end_date, interval_level='year'):

    dates = []

    start_date = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
    end_date = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

    if interval_level == 'year':
        years = relativedelta(end_date, start_date).years
        start_year = start_date.year

        for i in range(years+1):
            date = datetime(start_year+i, 1, 1)

            if date < end_date and date > start_date:
                dates.append(date)

        dates = [start_date] + dates + [end_date]

    elif interval_level == 'month':
        months = relativedelta(end_date, start_date).years * 12 + relativedelta(end_date, start_date).months

        start_month = datetime(start_date.year, start_date.month, 1)
        for i in range(months+1):
            date = start_month + relativedelta(months=i)

            if date < end_date and date > start_date:
                dates.append(date)

        dates = [start_date] + dates + [end_date]

    elif interval_level == 'day':
        days = (end_date - start_date).days
        for i in range(days+1):
            date = start_date + relativedelta(days=i)

            if date < end_date and date > start_date:
                dates.append(date)

        dates = [start_date] + dates + [end_date]

    intervals = []

    for i in range(len(dates)-1):
        item = {
            "name": f"{dates[i].strftime('%Y-%m-%d')}_to_{dates[i+1].strftime('%Y-%m-%d')}",
            "start": dates[i],
            "end": dates[i+1]
        }
        intervals.append(item)

    return intervals

# intervals = generate_intervals(start_date="1994-03-01", end_date="2024-03-01", interval_level='year')
# print(intervals)

# intervals = generate_intervals(start_date="1994-03-01", end_date="2024-03-01", interval_level='month')
# print(intervals)

# intervals = generate_intervals(start_date="1994-03-01", end_date="2024-03-01", interval_level='day')
# print(intervals)

def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root

def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, logger, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * epoch / cfg.warmup_epochs
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - cfg.warmup_epochs)
                / (cfg.epochs - cfg.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def get_cpu_usage():
    """Get CPU usage percentage"""
    return psutil.cpu_percent(interval=1)


def get_cpu_memory_usage():
    """Get memory usage percentage"""
    memory = psutil.virtual_memory()
    return memory.percent

def get_gpu_usage():
    """Get total GPU utilization percentage"""
    if not torch.cuda.is_available():
        return 0

    total_gpu_utilization = 0
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        total_gpu_utilization += torch.cuda.utilization(i)

    return total_gpu_utilization / device_count


def get_gpu_memory_usage():
    """Get total GPU memory usage percentage and total memory used"""
    if not torch.cuda.is_available():
        return 0

    total_memory_used = 0
    total_memory_total = 0
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        total_memory_used += torch.cuda.memory_allocated(i)
        total_memory_total += torch.cuda.get_device_properties(i).total_memory

    total_memory_usage_percent = (total_memory_used / total_memory_total) * 100 if total_memory_total > 0 else 0
    return total_memory_usage_percent

def all_reduce_mean(x):
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
