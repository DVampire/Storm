import torch
from torch import nn

from storm.registry import LOSS_FUNC

@LOSS_FUNC.register_module(force=True)
class VQVAELoss(nn.Module):
    def __init__(self,
                 nll_loss_weight = 1.0):
        super().__init__()
        self.nll_loss_weight = nll_loss_weight

    def __str__(self):
        return f"VQVAELoss(nll_loss_weight={self.nll_loss_weight})"

    def forward(
        self,
        sample,
        target_sample,
        mask = None,
        if_mask = False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert sample.shape == target_sample.shape

        rec_loss = (sample - target_sample) ** 2
        nll_loss = rec_loss

        if if_mask:
            mask = mask.repeat(1, 1, nll_loss.shape[-1])
            nll_loss = nll_loss * mask

        # Weighted NLL Loss
        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
        )

        return loss_dict