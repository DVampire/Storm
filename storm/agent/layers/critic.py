import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from functools import partial
from typing import Tuple
from einops import rearrange

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from storm.agent.layers.modules import create_modules
from storm.agent.layers.custom import DiscreteEmbeddingLayer, AttentionFusion

class Critic(nn.Module):
    def __init__(self,
                 *args,
                 input_size: Tuple = (64, 128),
                 embed_dim: int = 256,
                 seq_len: int = 64,
                 feature_dim: int = 128,
                 action_dim: int = 3,
                 middle_dim: int = 8,
                 output_dim: int = 1,
                 group_size: int = 3,
                 depth: int = 2,
                 num_head: int = 8,
                 num_max_tokens: int = 1000,
                 num_block_tokens: int = 32,
                 flash: bool = True,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 method: Literal["mlp", "mlp_full", "transformer", "gpt", "mamba", "mamba2"] = "mlp",
                 **kwargs
                 ):
        super(Critic, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.middle_dim = middle_dim
        self.output_dim = output_dim
        self.group_size = group_size
        self.depth = depth
        self.num_head = num_head
        self.num_max_tokens = num_max_tokens
        self.num_block_tokens = num_block_tokens
        self.total_num_block_tokens = num_block_tokens * 3  # features + cashes + positions + actions + rets
        self.flash = flash
        self.norm_layer = norm_layer
        self.method = method

        modules = create_modules(
            input_size=input_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            action_dim=action_dim,
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

        self.encoder_layer = modules['encoder_layer']
        self.blocks = modules['blocks']
        self.norm = modules['norm']
        self.middle_layer = modules['middle_layer']
        self.decoder_layer = modules['decoder_layer']

        self.action_encoder = DiscreteEmbeddingLayer(
            num_max_tokens=action_dim,
            embed_dim=embed_dim,
        )

        self.atten_fusion = AttentionFusion(
            state_dim=embed_dim,
            action_dim=embed_dim,
            embed_dim=embed_dim,
        )

        self.critic_blocks = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm_decoder = norm_layer(embed_dim)

    def forward_encoder(self, x: TensorDict):
        features = x["features"]
        cashes = x["cashes"]
        positions = x["positions"]
        actions = x["actions"]
        rets = x["rets"]

        if self.method == "mlp":
            stem_layer = self.encoder_layer['stem_layer']
            x = stem_layer(features)

            x = self.blocks(x) # extract features
            x = self.norm(x) # normalize

            x = rearrange(x, '... d n -> ... n d') # rearrange dimensions
            reduce_seq_layer = self.middle_layer['reduce_seq_layer']
            x = reduce_seq_layer(x) # reduce sequence dimension
            x = rearrange(x, '... n d -> ... (d n)') # flatten (..., sequence, embedding) to (..., sequence * embedding)

            reduce_embed_layer = self.middle_layer['reduce_embed_layer']
            x = reduce_embed_layer(x) # reduce embedding dimension

        elif self.method == "mlp_full":
            b, n = features.shape[:2]

            if len(features.shape) > 4:
                features = rearrange(features, 'b n ... -> (b n) ...')
                cashes = rearrange(cashes, 'b n ... -> (b n) ...')
                positions = rearrange(positions, 'b n ... -> (b n) ...')

                features_embedding = self.encoder_layer['features'](features)
                cashes_embedding = self.encoder_layer['cashes'](cashes)
                positions_embedding = self.encoder_layer['positions'](positions.long())

                state_embedding = torch.cat([features_embedding, cashes_embedding, positions_embedding], dim=-1)

                x = self.blocks(state_embedding)  # extract features
                x = self.norm(x)  # normalize

                x = rearrange(x, '(b n) ... -> b n ...', b=b, n=n)

            else:
                features_embedding = self.encoder_layer['features'](features)
                cashes_embedding = self.encoder_layer['cashes'](cashes)
                positions_embedding = self.encoder_layer['positions'](positions.long())

                state_embedding = torch.cat([features_embedding, cashes_embedding, positions_embedding], dim=-1)

                x = self.blocks(state_embedding)
                x = self.norm(x)

            x = rearrange(x, '... d n -> ... n d')  # rearrange dimensions
            reduce_seq_layer = self.middle_layer['reduce_seq_layer']
            x = reduce_seq_layer(x)  # reduce sequence dimension
            x = rearrange(x,'... n d -> ... (d n)')  # flatten (..., sequence, embedding) to (..., sequence * embedding)

            reduce_embed_layer = self.middle_layer['reduce_embed_layer']
            x = reduce_embed_layer(x)  # reduce embedding dimension

        elif self.method == "transformer":
            stem_layer = self.encoder_layer['stem_layer']
            x = stem_layer(features)

            indices_layer = self.encoder_layer['indices']
            indices_embedding = indices_layer(torch.arange(self.seq_len).to(x.device)) # positiona embedding
            x = x + indices_embedding # add position embedding

            if len(x.shape) > 3:
                b, n = x.shape[:2]
                x = rearrange(x, 'b n ... -> (b n) ...', b=b, n=n)
                x = self.blocks(x) # extract features
                x = rearrange(x, '(b n) ... -> b n ...', b=b, n=n)
            else:
                x = self.blocks(x)

            x = self.norm(x)

            x = x[..., -1, :] # get last token

        elif self.method == "gpt" or self.method == "mamba" or self.method == "mamba2":

            if len(features.shape) > 4:
                b, n = features.shape[:2]

                features = rearrange(features, 'b n ... -> (b n) ...')
                cashes = rearrange(cashes, 'b n ... -> (b n) ...')
                positions = rearrange(positions, 'b n ... -> (b n) ...')
                actions = rearrange(actions, 'b n ... -> (b n) ...')
                rets = rearrange(rets, 'b n ... -> (b n) ...')

                features_embedding = self.encoder_layer['features'](features)
                cashes_embedding = self.encoder_layer['cashes'](cashes)
                positions_embedding = self.encoder_layer['positions'](positions.long())
                actions_embedding = self.encoder_layer['actions'](actions.long())
                rets_embedding = self.encoder_layer['rets'](rets)

                indices_embedding = self.encoder_layer['indices'](torch.arange(self.total_num_block_tokens).to(features.device))

                state_embedding = torch.cat([features_embedding, cashes_embedding, positions_embedding], dim=-1)
                state_embedding = torch.stack([state_embedding,
                                        actions_embedding,
                                        rets_embedding,
                                 ], dim=1).view(b * n, -1, self.embed_dim * 2)

                x = state_embedding + indices_embedding

                x = self.blocks(x)
                x = self.norm(x)

                x = rearrange(x, '(b n) ... -> b n ...', b=b, n=n)
            else:
                features_embedding = self.encoder_layer['features'](features)
                cashes_embedding = self.encoder_layer['cashes'](cashes)
                positions_embedding = self.encoder_layer['positions'](positions.long())
                actions_embedding = self.encoder_layer['actions'](actions.long())
                rets_embedding = self.encoder_layer['rets'](rets)
                indices_embedding = self.encoder_layer['indices'](torch.arange(self.total_num_block_tokens).to(features.device))

                state_embedding = torch.cat([features_embedding, cashes_embedding, positions_embedding], dim=-1)
                state_embedding = torch.stack([state_embedding,
                                        actions_embedding,
                                        rets_embedding,
                                 ], dim=1).view(features.shape[0], -1, self.embed_dim * 2)

                x = state_embedding + indices_embedding

                x = self.blocks(x)
                x = self.norm(x)

            reduce_embed_layer = self.middle_layer['reduce_embed_layer']
            x = reduce_embed_layer(x) # reduce embedding dimension

            x = x[..., -3, :] # get last state token embedding

        return x

    def decoder(self, x: Tensor, a: Tensor = None):
        if a is not None:
            action_embedding = self.action_encoder(a)
            action_embedding = action_embedding[..., -1, :]

            x = self.atten_fusion(x, action_embedding)

        x = self.critic_blocks(x)
        x = self.norm_decoder(x)

        x = self.decoder_layer(x)
        return x

    def forward(self, x: TensorDict, a: Tensor = None):
        latent = self.forward_encoder(x)
        pred = self.decoder(latent, a).squeeze(-1)
        return pred

if __name__ == '__main__':

    device = torch.device("cpu")

    #########################mlp model#########################
    model = Critic(
        input_size=(32, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 1,
        method = "mlp",
    )
    # print(model)
    batch = TensorDict({
        "features": torch.randn(4, 32, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)
    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    action = torch.randint(0, 3, (4, 4, 32))
    pred = model(batch, action)
    print(pred.shape)
    #########################mlp model#########################

    print("-"*50)

    #########################transformer model#########################
    model = Critic(
        input_size=(32, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 1,
        method = "transformer",
    )
    # print(model)
    batch = TensorDict({
        "features": torch.randn(4, 32, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)
    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    action = torch.randint(0, 3, (4, 4, 32))
    pred = model(batch, action)
    print(pred.shape)
    #########################transformer model#########################

    print("-" * 50)

    #########################mlp full model#########################
    model = Critic(
        input_size=(32, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim=128,
        output_dim=1,
        method="mlp_full",
    )
    # print(model)
    batch = TensorDict({
        "features": torch.randn(4, 32, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)
    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)
    pred = model(batch)
    print(pred.shape)
    #########################mlp full model#########################

    print("-" * 50)

    #########################gpt model#########################
    model = Critic(
        input_size=(4, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 1,
        method = "gpt",
    )

    batch = TensorDict({
        "features": torch.randn(4, 32, 4, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)

    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 4, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)

    action = torch.randint(0, 3, (4, 4, 32))
    pred = model(batch, action)
    print(pred.shape)
    #########################gpt model#########################

    print("-" * 50)

    #########################mamba model#########################
    model = Critic(
        input_size=(4, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 1,
        method = "mamba",
    )

    batch = TensorDict({
        "features": torch.randn(4, 32, 4, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)

    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 4, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)

    action = torch.randint(0, 3, (4, 4, 32))
    pred = model(batch, action)
    print(pred.shape)
    #########################mamba model#########################

    print("-" * 50)

    #########################mamba2 model#########################

    model = Critic(
        input_size=(4, 152),
        feature_dim=152,
        seq_len=32,
        embed_dim = 128,
        output_dim = 1,
        method = "mamba2",
    )

    batch = TensorDict({
        "features": torch.randn(4, 32, 4, 152),
        "cashes": torch.randn(4, 32),
        "positions": torch.randint(0, 1000, (4, 32)),
        "actions": torch.randint(0, 3, (4, 32)),
        "rets": torch.randn(4, 32),
    }, batch_size=4)

    pred = model(batch)
    print(pred.shape)

    batch = TensorDict({
        "features": torch.randn(4, 4, 32, 4, 152),
        "cashes": torch.randn(4, 4, 32),
        "positions": torch.randint(0, 1000, (4, 4, 32)),
        "actions": torch.randint(0, 3, (4, 4, 32)),
        "rets": torch.randn(4, 4, 32),
    }, batch_size=4)

    action = torch.randint(0, 3, (4, 4, 32))
    pred = model(batch, action)
    print(pred.shape)
    #########################mamba2 model#########################