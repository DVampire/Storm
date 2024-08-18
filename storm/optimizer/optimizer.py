import torch
from transformers.optimization import AdamW
from colossalai.nn.optimizer import HybridAdam

from storm.registry import OPTIMIZER

OPTIMIZER.register_module(name='AdamW', module=AdamW)
OPTIMIZER.register_module(name='HybridAdam', module=HybridAdam)
