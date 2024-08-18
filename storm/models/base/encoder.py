import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError