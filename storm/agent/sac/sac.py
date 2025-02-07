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
class SAC(nn.Module):
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

        super(SAC, self).__init__()


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

        self.critic1 = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            group_size=group_size,
            output_dim=output_dim,
            depth=depth,
            num_head=num_head,
            num_max_tokens=num_max_tokens,
            num_block_tokens=num_block_tokens,
            flash=flash,
            norm_layer=norm_layer,
            method=method,
        )

        self.critic2 = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            group_size=group_size,
            output_dim=output_dim,
            depth=depth,
            num_head=num_head,
            num_max_tokens=num_max_tokens,
            num_block_tokens=num_block_tokens,
            flash=flash,
            norm_layer=norm_layer,
            method=method,
        )

        self.target_critic1 = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            group_size=group_size,
            output_dim=output_dim,
            depth=depth,
            num_head=num_head,
            num_max_tokens=num_max_tokens,
            num_block_tokens=num_block_tokens,
            flash=flash,
            norm_layer=norm_layer,
            method=method,
        )

        self.target_critic2 = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            group_size=group_size,
            output_dim=output_dim,
            depth=depth,
            num_head=num_head,
            num_max_tokens=num_max_tokens,
            num_block_tokens=num_block_tokens,
            flash=flash,
            norm_layer=norm_layer,
            method=method,
        )

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def get_value(self, x: TensorDict, a: Tensor = None, use_target: bool = False):
        if use_target:
            return self.target_critic1(x, a), self.target_critic2(x, a)
        return self.critic1(x, a), self.critic2(x, a)

    def get_action(self,
                   x: TensorDict,
                   ):

        logits = self.actor(x)

        if len(logits.shape) == 2:
            logits = logits.unsqueeze(1)

        b, c, n = logits.shape

        logits = rearrange(logits , "b c n -> (b c) n", b=b, c=c, n=n)

        dist = Categorical(logits=logits)

        action = dist.sample()
        action_probs = dist.probs
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)

        action = rearrange(action, "(b c) -> b c", b=b, c=c).squeeze(1)
        action_probs = rearrange(action_probs, "(b c) n -> b c n", b=b, c=c).squeeze(1)
        log_prob = rearrange(log_prob, "(b c) n -> b c n", b=b, c=c).squeeze(1)

        return action, log_prob, action_probs

    def forward(self, *input, **kwargs):
        pass

if __name__ == '__main__':
    device = torch.device("cpu")

    model = SAC(
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
    action, log_prob, action_probs = model.get_action(batch)
    print(action.shape, log_prob.shape, action_probs.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    action,  log_prob, action_probs = model.get_action(batch)
    print(action.shape, log_prob.shape, action_probs.shape)