import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from storm.registry import EMBED


@EMBED.register_module(force=True)
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        *args,
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel: int = 1,
        input_dim: int = 152,
        temporal_dim: int = 3,
        embed_dim: int = 128,
        if_use_stem: bool = False,
        stem_embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        data_size = to_2tuple(data_size)
        patch_size = to_2tuple(patch_size)
        assert (data_size[0] % patch_size[0] == 0 and data_size[1] % patch_size[1] == 0
                and data_size[2] % patch_size[2] == 0), f"Data size {data_size} must be divisible by patch size {patch_size}"

        self.input_channel = input_channel
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        self.data_size = data_size
        self.patch_size = patch_size

        self.p_size = (self.patch_size[0],
                       self.patch_size[1],
                       self.patch_size[2]) # p1, p2, p3
        self.p_num = self.p_size[0] * self.p_size[1] * self.p_size[2] # p1 * p2 * p3
        self.n_size = (self.data_size[0] // self.patch_size[0],
                       self.data_size[1] // self.patch_size[1],
                       self.data_size[2] // self.patch_size[2]) # n1, n2, n3
        self.n_num = self.n_size[0] * self.n_size[1] * self.n_size[2] # n1 * n2 * n3

        self.if_use_stem = if_use_stem
        self.stem_embedding_dim = stem_embedding_dim

        if self.if_use_stem:
            self.stem_layer = nn.Linear(self.data_size[-1], self.stem_embedding_dim)
            kernel_size = (patch_size[0], patch_size[1], self.stem_embedding_dim)
        else:
            kernel_size = patch_size

        self.proj = nn.Conv3d(
            in_channels=self.input_channel, out_channels = embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        N, C, T, S, F = x.shape # batch, channel, temporal, spatial, feature
        assert (T == self.data_size[0] and S == self.data_size[1]
                and F == self.data_size[2]), f"Input data size {(T, N, F)} doesn't match model {self.data_size}."

        if self.if_use_stem:
            x = self.stem_layer(x)

        x = self.proj(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)
        return x


if __name__ == '__main__':
    device = torch.device("cpu")

    model = PatchEmbed(data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_dim = 152,
        input_channel=1,
        temporal_dim= 3,
        embed_dim= 128,
        if_use_stem=True,
        stem_embedding_dim=64).to(device)

    print(model)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    emb = model(batch)
    print(emb.shape)