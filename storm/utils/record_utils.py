import numpy as np
import torch
from typing import Dict
from accelerate import Accelerator

class Records():
    def __init__(self, accelerator: Accelerator = None):

        self.accelerator = accelerator

        self.raw_item = dict()  # gathered before accelerator, TENSOR
        self.gathered_item = dict()  # gathered after accelerator, NOT TENSOR
        self.combiner = dict()  # combined after accelerator, NOT TENSOR

        self.extra_raw_item = dict()  # gathered before accelerator, TENSOR
        self.extra_gathered_item = dict()  # gathered after accelerator, NOT TENSOR
        self.extra_combiner = dict()  # combined after accelerator, NOT TENSOR

    def _gather_values(self, data: Dict[str, torch.Tensor], if_value = True):

        keys = list(data.keys())
        values = list(data.values())

        gathered_values = self.accelerator.gather(values)
        gathered_values = dict(zip(keys, gathered_values))

        gathered_dict = {}

        for key, value in gathered_values.items():
            if if_value:
                gathered_value = value.mean().item()
            else:
                gathered_value = value.cpu().numpy()

            gathered_dict[key] = gathered_value

        return gathered_dict

    def update_value(self, data: Dict[str, float]):
        self.gathered_item.update(data)

    def update(self,
               data: Dict[str, torch.Tensor] = None,
               extra_info: Dict[str, np.ndarray] = None
               ):
        """
        update the raw item and extra combiner
        :param data:
        :param extra_info:
        :return:
        """
        if data is not None:
            self.raw_item.update(data)
        if extra_info is not None:
            self.extra_raw_item.update(extra_info)

    def gather(self, train_gather_multi_gpu: bool = False):
        """
        gather the values
        :return:
        """
        if train_gather_multi_gpu:
            gathered_item = self._gather_values(self.raw_item, if_value=True)
        else:
            gathered_item = {
                key: value.mean().item() for key, value in self.raw_item.items()
            }

        self.gathered_item.update(gathered_item)
        for key, value in self.gathered_item.items():
            self.combiner.setdefault(key, []).append(value)

        if train_gather_multi_gpu:
            extra_gathered_item = self._gather_values(self.extra_raw_item, if_value=False)
        else:
            extra_gathered_item = {
                key: value.cpu().numpy() for key, value in self.extra_raw_item.items()
            }
        self.extra_gathered_item.update(extra_gathered_item)
        for key, value in self.extra_gathered_item.items():
            self.extra_combiner.setdefault(key, []).append(value)

        self.raw_item = dict()
        self.extra_raw_item = dict()