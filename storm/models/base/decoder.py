import torch.nn as nn

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
