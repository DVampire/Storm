import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import Mlp

class Critic(nn.Module):
    def __init__(self,
                 *args,
                 input_size = (64, 128),
                 embed_dim: int = 256,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 output_dim = 3,
                 **kwargs
                 ):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.output_dim = output_dim

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.encoder_layer = nn.Linear(
            self.input_size[1],
            embed_dim,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim,
                    hidden_features=embed_dim,
                    act_layer=nn.Tanh,
                    out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_layer = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

        self.proj = nn.Linear(
            self.input_size[0],
            self.output_dim,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):

        x = self.encoder_layer(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_layer(x).squeeze(-1)
        x = self.proj(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent).squeeze(-1)
        return pred

if __name__ == '__main__':

    device = torch.device("cpu")

    model = Critic(
        input_size=(512, 128),
        embed_dim=128,
        cls_embed=False,
        output_dim=1,
    )

    batch = torch.randn(4, 512, 128)
    pred = model(batch)
    print(pred.shape)

    batch = torch.randn(4, 4, 512, 128)
    pred = model(batch)
    print(pred.shape)