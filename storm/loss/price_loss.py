from torch import nn
import torch.nn.functional as F
from einops import rearrange

from storm.registry import LOSS_FUNC

@LOSS_FUNC.register_module(force=True)
class PriceConstraintEntropyLoss(nn.Module):
    def __init__(self, cont_loss_weight = 1.0):
        super().__init__()
        self.cont_loss_weight = cont_loss_weight

    def __str__(self):
        return f"PriceConstraintEntropyLoss(cont_loss_weight={self.cont_loss_weight})"

    def forward(self, prices):

        """
        :param prices: (N, T, S, 5)
        :return:
        """
        prices = rearrange(prices, 'n t s c -> (n t s) c')

        # open, high, low, close, adj_close
        low = prices[..., 2].unsqueeze(1)
        others = prices[..., [0, 1, 3, 4]]
        lower_loss = F.relu(low - others).mean()

        high = prices[..., 1].unsqueeze(1)
        others = prices[..., [0, 2, 3, 4]]
        upper_loss = F.relu(others - high).mean()

        constraint_loss = upper_loss + lower_loss
        constraint_loss = constraint_loss.mean()

        weighted_cont_loss = constraint_loss
        if self.cont_loss_weight is not None and self.cont_loss_weight > 0.0:
            weighted_cont_loss = self.cont_loss_weight * constraint_loss

        loss_dict = {
            "weighted_cont_loss": weighted_cont_loss
        }

        return loss_dict