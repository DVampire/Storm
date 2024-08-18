import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from torch.distributions.categorical import Categorical

from storm.registry import AGENT
from storm.agent.actors import Actor
from storm.agent.critics import Critic

@AGENT.register_module(force=True)
class PPO(nn.Module):
    def __init__(self,
                 *args,
                 input_size=(64, 128),
                 embed_dim: int = 256,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 action_dim=3,
                 **kwargs
                 ):

        super(PPO, self).__init__()

        self.actor = Actor(
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
        )

        self.critic = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=1,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        if len(logits .shape) == 2:
            logits  = logits.unsqueeze(1)

        b, c, n = logits.shape

        logits = rearrange(logits , "b c n -> (b c) n", b=b, c=c, n=n)

        dis = Categorical(logits=logits)

        if action is None:
            action = dis.sample()

        probs = dis.log_prob(action)
        entropy = dis.entropy()
        value = self.critic(x)

        action = rearrange(action, "(b c) -> b c", b=b, c=c).squeeze(1)
        probs = rearrange(probs, "(b c) -> b c", b=b, c=c).squeeze(1)
        entropy = rearrange(entropy, "(b c) -> b c", b=b, c=c).squeeze(1)

        return action, probs, entropy, value

    def forward(self, *input, **kwargs):
        pass

if __name__ == '__main__':
    device = torch.device("cpu")

    model = PPO(
        input_size=(512, 128),
        embed_dim=128,
        cls_embed=False,
        action_dim=3,
    ).to(device)

    batch = torch.randn(4, 512, 128)
    action, probs, entropy, value = model.get_action_and_value(batch)
    print(action.shape, probs.shape, entropy.shape, value.shape)