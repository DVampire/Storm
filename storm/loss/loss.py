import torch
import torch.nn.functional as F

from storm.registry import LOSS_FUNC

@LOSS_FUNC.register_module(force=True)
def cross_entropy(*args,
                  logits,
                  target,
                  ignore_index=-1,
                  **kwargs):
    loss = F.cross_entropy(logits, target, ignore_index=ignore_index)

    loss_dict = {
        "loss": loss
    }

    return loss_dict

@LOSS_FUNC.register_module(force=True)
def base_loss(*args,
              pred_prices,
              target_prices,
              restored_pred_prices,
              restored_target_prices,
              mask = None,
              if_mask = False,
              **kwargs):

    rec_loss = (pred_prices - target_prices) ** 2
    rec_loss = rec_loss.mean(dim=-1)  # [N, L], mean loss per patch

    if if_mask:
        mask = mask.view(rec_loss.shape)
        rec_loss = (rec_loss * mask).sum() / mask.sum()  # mean loss on removed patches
    else:
        rec_loss = rec_loss.mean()

    loss_dict = {
        "rec_loss": rec_loss,
    }

    return loss_dict

@LOSS_FUNC.register_module(force=True)
def ohlc_constraint_loss(*args,
                         pred_prices,
                         target_prices,
                         restored_pred_prices,
                         restored_target_prices,
                         mask=None,
                         if_mask=False,
                         **kwargs):

    restored_pred_prices = restored_pred_prices.reshape((-1, restored_pred_prices.shape[-1]))

    # open, high, low, close, adj_close
    low = restored_pred_prices[..., 2].unsqueeze(1)
    others = restored_pred_prices[..., [0, 1, 3, 4]]
    lower_loss = F.relu(low - others).mean()

    high = restored_pred_prices[..., 1].unsqueeze(1)
    others = restored_pred_prices[..., [0, 2, 3, 4]]
    upper_loss = F.relu(others - high).mean()

    constraint_loss = upper_loss + lower_loss
    constraint_loss = constraint_loss.mean()

    rec_loss = (pred_prices - target_prices) ** 2
    rec_loss = rec_loss.mean(dim=-1)  # [N, L], mean loss per patch

    if if_mask:
        mask = mask.view(rec_loss.shape)
        rec_loss = (rec_loss * mask).sum() / mask.sum()  # mean loss on removed patches
    else:
        rec_loss = rec_loss.mean()

    loss_dict = {
        "rec_loss": rec_loss,
        "cont_loss": constraint_loss
    }

    return loss_dict

@LOSS_FUNC.register_module(force=True)
def ohlc_constraint_entropy_loss(*args,
                                 pred_prices,
                                 target_prices,
                                 restored_pred_prices,
                                 restored_target_prices,
                                 mask=None,
                                 if_mask=False,
                                 **kwargs):

    restored_pred_prices = restored_pred_prices.reshape((-1, restored_pred_prices.shape[-1]))
    restored_target_prices = restored_target_prices.reshape((-1, restored_target_prices.shape[-1]))

    min_values, min_labels = restored_target_prices.min(dim=1)
    max_values, max_labels = restored_target_prices.max(dim=1)

    lower_loss = F.cross_entropy(-restored_pred_prices, min_labels)
    upper_loss = F.cross_entropy(restored_pred_prices, max_labels)

    constraint_loss = upper_loss + lower_loss

    rec_loss = (pred_prices - target_prices) ** 2
    rec_loss = rec_loss.mean(dim=-1)  # [N, L], mean loss per patch

    if if_mask:
        mask = mask.view(rec_loss.shape)
        rec_loss = (rec_loss * mask).sum() / mask.sum()  # mean loss on removed patches
    else:
        rec_loss = rec_loss.mean()

    loss_dict = {
        "rec_loss": rec_loss,
        "cont_loss": constraint_loss
    }

    return loss_dict

if __name__ == '__main__':
    loss_func = LOSS_FUNC.get("base_loss")
    print(loss_func)