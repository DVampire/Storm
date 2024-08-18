import torch
from typing import List

def RankIC(preds: torch.Tensor, actuals: torch.Tensor):

    preds = preds.view(-1)
    actuals = actuals.view(-1)

    preds_rank = preds.argsort().argsort().float()
    actuals_rank = actuals.argsort().argsort().float()

    covariance = torch.mean((preds_rank - preds_rank.mean()) * (actuals_rank - actuals_rank.mean()))
    preds_std = preds_rank.std()
    actuals_std = actuals_rank.std()

    rank_ic = covariance / (preds_std * actuals_std)
    return rank_ic


def RankICIR(rank_ic_values: List[torch.Tensor]):

    if len(rank_ic_values) <= 1:
        return torch.tensor(0.0).to(rank_ic_values[0].device)

    mean_rank_ic = torch.mean(torch.stack(rank_ic_values))
    std_rank_ic = torch.std(torch.stack(rank_ic_values))

    rank_ic_ir = mean_rank_ic / (std_rank_ic + 1e-6)
    return rank_ic_ir

if __name__ == '__main__':
    preds = torch.tensor([0.3, 0.2, 0.9, 0.7, 0.1])
    actuals = torch.tensor([0.4, 0.1, 0.8, 0.6, 0.2])

    rank_ic = RankIC(preds, actuals)
    print(f"RankIC: {rank_ic.item()}")

    rank_ic_values = [rank_ic, rank_ic, rank_ic]
    rank_ic_ir = RankICIR(rank_ic_values)
    print(f"RankICIR: {rank_ic_ir.item()}")