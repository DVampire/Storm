import torch
import torch.nn as nn
from functools import partial
from typing import Tuple
from tensordict import TensorDict
from torch import Tensor
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from storm.registry import AGENT
from storm.agent.layers.actor import Actor

@AGENT.register_module(force=True)
class DPO(nn.Module):
    def __init__(self,
                 *args,
                 input_size: Tuple = (64, 128),
                 embed_dim: int = 256,
                 seq_len: int = 64,
                 feature_dim: int = 128,
                 action_dim: int = 3,
                 middle_dim: int = 8,
                 output_dim: int = 3,
                 group_size: int = 3,
                 depth: int = 2,
                 num_head: int = 8,
                 num_max_tokens: int = 1000,
                 num_block_tokens: int = 32,
                 flash: bool = True,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 method: Literal["mlp", "transformer", "gpt", "mamba", "mamba2"] = "mlp",
                 **kwargs
                 ):

        super(DPO, self).__init__()

        self.actor = Actor(
            input_size=input_size,
            embed_dim=embed_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            output_dim=output_dim,
            group_size=group_size,
            depth=depth,
            num_head=num_head,
            num_max_tokens=num_max_tokens,
            num_block_tokens=num_block_tokens,
            flash=flash,
            norm_layer=norm_layer,
            method=method,
        )

    def get_action(self, x: TensorDict) -> Tuple[Tensor, Tensor]:
        probs = self.forward(x)
        actions = torch.argmax(probs, dim=-1)
        return actions, probs

    def forward(self, x: TensorDict) -> Tensor:
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == '__main__':
    device = torch.device("cpu")

    model = DPO(
        input_size=(32, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 3,
        method = "mlp",
    ).to(device)

    batch = TensorDict({
        "features": torch.randn(4, 32, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)
    probs = model(batch)
    print(probs.shape)
    actions = model.get_action(batch)
    print(actions.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    probs = model(batch)
    print(probs.shape)
    actions = model.get_action(batch)
    print(actions.shape)