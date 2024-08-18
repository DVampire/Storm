import torch
import torch.nn.functional as F

def MSE(y_true, y_pred):
    if len(y_true.shape) >= 3:
        y_true = y_true.view(-1, y_true.shape[-1])
    if len(y_pred.shape) >= 3:
        y_pred = y_pred.view(-1, y_pred.shape[-1])

    # process nan and inf
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    return F.mse_loss(y_true, y_pred)

if __name__ == '__main__':
    # Example usage
    y_true = torch.tensor([[1.0, 2.0], [3.0, float('nan')], [float('inf'), 5.0]])
    y_pred = torch.tensor([[1.0, 2.1], [2.9, 3.0], [4.5, float('-inf')]])

    mse = MSE(y_true, y_pred)
    print(mse)