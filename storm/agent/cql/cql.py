import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from torch import Tensor
from tensordict import TensorDict
from typing import Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from storm.registry import AGENT
from storm.agent.layers.actor import Actor
from storm.agent.layers.critic import Critic
from torch.distributions.categorical import Categorical

@AGENT.register_module(force=True)
class CQL(nn.Module):
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

        super(CQL, self).__init__()

        self.critic = Critic(
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

        self.target_critic = Critic(
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

        self.target_critic.load_state_dict(self.critic.state_dict())


    def forward(self, x: TensorDict, use_target: bool = False) -> Tensor:

        if use_target:
            logits = self.target_critic(x)
        else:
            logits = self.critic(x)

        return logits

if __name__ == '__main__':
    device = torch.device("cpu")

    model = CQL(
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
    logits = model(batch)
    print(logits.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    logits = model(batch)
    print(logits.shape)