import torch
from torch import nn as nn
from timm.models.layers import to_2tuple
from einops import rearrange

from storm.registry import ENCODER
from storm.registry import EMBED
from storm.models import BaseEncoder
from storm.models.modules.transformer import Mlp

@ENCODER.register_module(force=True)
class FactorVAEEncoder(BaseEncoder):
    def __init__(self,
                 embed_config: dict = None,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 portfolio_num: int = 20,
                 factor_num: int = 32,
                 trunc_init: bool = False):
        super(FactorVAEEncoder, self).__init__()

        self.data_size = to_2tuple(embed_config.get('data_size', None))
        self.patch_size = to_2tuple(embed_config.get('patch_size', None))

        self.input_size = (
            self.data_size[0] // self.patch_size[0],
            self.data_size[1] // self.patch_size[1],
            self.data_size[2] // self.patch_size[2]
        )
        self.num_patches = self.input_size[0] * self.input_size[1] * self.input_size[2]

        self.trunc_init = trunc_init
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.portfolio_layer = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim,
            out_features=portfolio_num,
            act_layer=nn.LeakyReLU,
        )

        self.post_mu = Mlp(
            in_features=portfolio_num,
            hidden_features=latent_dim,
            out_features=factor_num,
            act_layer=nn.LeakyReLU
        )

        self.post_sigma = Mlp(
            in_features=portfolio_num,
            hidden_features=latent_dim,
            out_features=factor_num,
            act_layer=nn.LeakyReLU
        )

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.initialize_weights()

    def initialize_weights(self):

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self,
                latent_features: torch.FloatTensor,
                label: torch.FloatTensor):

        N, _, NT, _ = label.shape

        label = rearrange(label, 'n c t s-> (n c t) s')

        latent_features = self.portfolio_layer(latent_features)
        latent_features = latent_features.permute(0, 2, 1)

        attn = self.softmax(latent_features)

        label = label.unsqueeze(-1)

        # yp = [batch, portfolio_num]
        yp = torch.matmul(attn, label).squeeze(-1)

        # mu = [batch, factor_num, 1]
        mu_post = self.post_mu(yp)
        mu_post = mu_post.unsqueeze(-1)

        # sigma = [batch, factor_num, 1]
        sigma_post = self.post_sigma(yp)
        sigma_post = self.softplus(sigma_post)
        sigma_post = sigma_post.unsqueeze(-1) + 1e-6

        return mu_post, sigma_post


if __name__ == '__main__':
    device = torch.device("cpu")

    embed_config = dict(
        type='FactorVAEEmbed',
        data_size=(64, 29, 152),
        patch_size=(64, 1, 152),
        input_dim=152,
        input_channel=1,
        temporal_dim=3,
        embed_dim=128)

    model = FactorVAEEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        portfolio_num=20,
        factor_num=32).to(device)

    embed_layer = EMBED.build(embed_config).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    label = torch.randn(4, 1, 1, 29) # batch, channel, next returns, asset nums

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    embed = embed_layer(batch)
    mu_post, sigma_post = model(embed, label)
    print(mu_post.shape, sigma_post.shape)