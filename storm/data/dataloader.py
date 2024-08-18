import torch
from typing import Any
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from accelerate import Accelerator

def prepare_dataloader(
    accelerator: Accelerator,
    collate_fn: Any,
    dataset: Dataset,
    batch_size: int = 256,
    shuffle = False,
    drop_last = False,
    pin_memory = False,
    num_workers = 0,
    distributed=False,
    train = True,
    **kwargs,
):
    r"""
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.


    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if train:
        if distributed:
            num_tasks = accelerator.num_processes  # world_size
            global_rank = accelerator.process_index  # global_rank

            sampler = DistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **_kwargs,
    )

    return dataloader

if __name__ == '__main__':
    from storm.registry import DATASET

    dataset = dict(
        type="PriceTextDataset",
        data_path="datasets/processd_day_dj30/features",
        assets_path="configs/_asset_list_/dj30.json",
        fields_name={
            "features": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "kmid",
                "kmid2",
                "klen",
                "kup",
                "kup2",
                "klow",
                "klow2",
                "ksft",
                "ksft2",
                "roc_5",
                "roc_10",
                "roc_20",
                "roc_30",
                "roc_60",
                "ma_5",
                "ma_10",
                "ma_20",
                "ma_30",
                "ma_60",
                "std_5",
                "std_10",
                "std_20",
                "std_30",
                "std_60",
                "beta_5",
                "beta_10",
                "beta_20",
                "beta_30",
                "beta_60",
                "max_5",
                "max_10",
                "max_20",
                "max_30",
                "max_60",
                "min_5",
                "min_10",
                "min_20",
                "min_30",
                "min_60",
                "qtlu_5",
                "qtlu_10",
                "qtlu_20",
                "qtlu_30",
                "qtlu_60",
                "qtld_5",
                "qtld_10",
                "qtld_20",
                "qtld_30",
                "qtld_60",
                "rank_5",
                "rank_10",
                "rank_20",
                "rank_30",
                "rank_60",
                "imax_5",
                "imax_10",
                "imax_20",
                "imax_30",
                "imax_60",
                "imin_5",
                "imin_10",
                "imin_20",
                "imin_30",
                "imin_60",
                "imxd_5",
                "imxd_10",
                "imxd_20",
                "imxd_30",
                "imxd_60",
                "rsv_5",
                "rsv_10",
                "rsv_20",
                "rsv_30",
                "rsv_60",
                "cntp_5",
                "cntp_10",
                "cntp_20",
                "cntp_30",
                "cntp_60",
                "cntn_5",
                "cntn_10",
                "cntn_20",
                "cntn_30",
                "cntn_60",
                "cntd_5",
                "cntd_10",
                "cntd_20",
                "cntd_30",
                "cntd_60",
                "corr_5",
                "corr_10",
                "corr_20",
                "corr_30",
                "corr_60",
                "cord_5",
                "cord_10",
                "cord_20",
                "cord_30",
                "cord_60",
                "sump_5",
                "sump_10",
                "sump_20",
                "sump_30",
                "sump_60",
                "sumn_5",
                "sumn_10",
                "sumn_20",
                "sumn_30",
                "sumn_60",
                "sumd_5",
                "sumd_10",
                "sumd_20",
                "sumd_30",
                "sumd_60",
                "vma_5",
                "vma_10",
                "vma_20",
                "vma_30",
                "vma_60",
                "vstd_5",
                "vstd_10",
                "vstd_20",
                "vstd_30",
                "vstd_60",
                "wvma_5",
                "wvma_10",
                "wvma_20",
                "wvma_30",
                "wvma_60",
                "vsump_5",
                "vsump_10",
                "vsump_20",
                "vsump_30",
                "vsump_60",
                "vsumn_5",
                "vsumn_10",
                "vsumn_20",
                "vsumn_30",
                "vsumn_60",
                "vsumd_5",
                "vsumd_10",
                "vsumd_20",
                "vsumd_30",
                "vsumd_60",
            ],
            "prices": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
            ],
            "temporals": [
                "day",
                "weekday",
                "month",
            ],
            "labels": [
                "ret1",
                "mov1"
            ]
        },
        if_norm=True,
        if_norm_temporal=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        scaler_file="scalers.joblib",
        scaled_data_file="scaled_data.joblib",
        history_timestamps=64,
        future_timestamps=32,
        patch_timestamps=4,
        start_timestamp="1994-03-01",
        end_timestamp="2024-03-01",
        timestamp_format="%Y-%m-%d",
        workdir="workdir",
        tag="storm",
    )

    dataset = DATASET.build(dataset)
    print(dataset)

