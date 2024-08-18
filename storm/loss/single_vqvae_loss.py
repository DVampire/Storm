import torch
from torch import nn

from storm.registry import LOSS_FUNC

@LOSS_FUNC.register_module(force=True)
class SingleVQVAELoss(nn.Module):
    def __init__(self,
                 cs_scale = 1.0,
                 nll_loss_weight = 1.0,
                 ret_loss_weight = 1.0,
                 kl_loss_weight = 0.000001):
        super().__init__()
        self.cs_scale = cs_scale
        self.nll_loss_weight = nll_loss_weight
        self.ret_loss_weight = ret_loss_weight
        self.kl_loss_weight = kl_loss_weight

    def __str__(self):
        return f"SingleVAELoss(cs_scale = {self.cs_scale}, nll_loss_weight={self.nll_loss_weight}, ret_loss_weight={self.ret_loss_weight}, kl_loss_weight={self.kl_loss_weight})"

    def forward(
        self,
        sample,
        target_sample,
        label,
        pred_label,
        posterior,
        prior,
        mask=None,
        if_mask=False,
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

        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        ret_loss = (label - pred_label) ** 2
        ret_loss = ret_loss.mean(dim=-1)

        weighted_ret_loss = ret_loss
        if self.ret_loss_weight is not None:
            weighted_ret_loss = self.ret_loss_weight * ret_loss
        weighted_ret_loss = torch.sum(weighted_ret_loss) / weighted_ret_loss.shape[0]

        kl_loss = posterior.kl(prior, dims=[1])
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = self.kl_loss_weight * kl_loss
        weighted_kl_loss = torch.sum(weighted_kl_loss) / weighted_kl_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_ret_loss=weighted_ret_loss,
            weighted_kl_loss=weighted_kl_loss,
        )

        return loss_dict