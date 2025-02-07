import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import Mlp
from typing import Tuple

try:
    from typing import Literal
    from mamba_ssm import Mamba, Mamba2
except ImportError:
    from typing_extensions import Literal

from storm.agent.layers.transformer import Block as TransformerBlock, GPTConfig
from storm.agent.layers.custom import FeaturesEmbedLayer, ContinuousEmbeddingLayer, DiscreteEmbeddingLayer


# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0, 0.02)

def create_modules(input_size: Tuple = (64, 128),
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
                   **kwargs):
    """
    Create a neural network module based on the provided configuration.

    Args:
        input_size: Tuple of (sequence length, feature dimension).
        embed_dim: Embedding dimension.
        seq_len: Sequence length.
        feature_dim: Number of input features per time step.
        action_dim: Action dimension for output.
        middle_dim: Hidden dimension for MLP intermediate layers.
        output_dim: Output dimension of the model.
        group_size: Group size for Transformer blocks.
        depth: Number of layers in the model.
        num_head: Number of attention heads for Transformer.
        flash: Whether to enable flash attention (if supported).
        norm_layer: Normalization layer.
        method: Type of model architecture ("mlp", "transformer", etc.).

    Returns:
        nn.Module: Configured neural network model.
    """

    # Stem layer to project feature_dim to embed_dim
    stem_layer = nn.Linear(feature_dim, embed_dim, bias=True)

    encoder_layer = None
    blocks = None
    norm = None
    middle_layer = None
    decoder_layer = None

    if method == 'mlp':
        encoder_layer = nn.ModuleDict(
            dict(
                stem_layer=stem_layer,
            )
        )

        blocks = nn.Sequential(*[
            Mlp(in_features=embed_dim,
                hidden_features=embed_dim,
                act_layer=nn.Tanh,
                out_features=embed_dim)
            for _ in range(depth)
        ])

        norm = norm_layer(embed_dim)

        middle_layer = nn.ModuleDict(
            dict(
                reduce_seq_layer=nn.Linear(seq_len, middle_dim),
                reduce_embed_layer=nn.Linear(middle_dim * embed_dim, embed_dim),
            )
        )

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    elif method == 'transformer':

        indices_embedding_layer = DiscreteEmbeddingLayer(
            num_max_tokens=seq_len,
            embed_dim=embed_dim,
        )

        encoder_layer = nn.ModuleDict(
            dict(
                indices=indices_embedding_layer,
                stem_layer=stem_layer,
            )
        )

        transformer_config = GPTConfig(
            embed_dim=embed_dim,
            num_head=num_head,
            dropout=0.0,
            bias=True,
            block_size=seq_len,
            group_size=group_size,
            flash=flash
        )

        blocks = nn.Sequential(*[
            TransformerBlock(transformer_config) for _ in range(depth)
        ])

        norm = norm_layer(embed_dim)

        middle_layer = nn.Identity()

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    elif method == "mlp_full":
        features_embed_layer = stem_layer

        cashes_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=8,
        )

        positions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=num_max_tokens,
            embed_dim=8,
        )

        encoder_layer = nn.ModuleDict(
            dict(
                features=features_embed_layer,
                cashes=cashes_embed_layer,
                positions=positions_embed_layer,  # position of the user in trading
            )
        )

        blocks = nn.Sequential(*[
            Mlp(in_features=embed_dim + 16,
                hidden_features=embed_dim + 16,
                act_layer=nn.Tanh,
                out_features=embed_dim + 16)
            for _ in range(depth)
        ])

        norm = norm_layer(embed_dim + 16)

        middle_layer = nn.ModuleDict(
            dict(
                reduce_seq_layer=nn.Linear(seq_len, middle_dim),
                reduce_embed_layer=nn.Linear(middle_dim * (embed_dim + 16), embed_dim),
            )
        )

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    elif method == 'gpt':

        features_embed_layer = FeaturesEmbedLayer(
            input_size=input_size,
            embed_dim=embed_dim,
            input_channel=1,
            depth=depth,
        )

        cashes_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim // 2,
        )

        positions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=num_max_tokens,
            embed_dim=embed_dim // 2,
        )

        actions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=action_dim,
            embed_dim=embed_dim * 2,
        )

        rets_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim * 2,
        )

        total_num_block_tokens = num_block_tokens * 3  # features + cashes + positions + actions + rets
        indices_embedding_layer = DiscreteEmbeddingLayer(
            num_max_tokens=total_num_block_tokens,
            embed_dim=embed_dim * 2,
        )

        encoder_layer = nn.ModuleDict(
            dict(
                features=features_embed_layer,
                cashes=cashes_embed_layer,
                positions=positions_embed_layer,  # position of the user in trading
                actions=actions_embed_layer,
                rets=rets_embed_layer,
                indices=indices_embedding_layer,  # position embedding of the sequence
            )
        )

        transformer_config = GPTConfig(
            embed_dim=embed_dim * 2,
            num_head=num_head,
            dropout=0.0,
            bias=True,
            block_size=total_num_block_tokens,
            group_size=group_size,
            flash=flash,
        )

        blocks = nn.Sequential(*[
            TransformerBlock(transformer_config) for _ in range(depth)
        ])

        norm = norm_layer(embed_dim * 2)

        middle_layer = nn.ModuleDict(
            dict(
                reduce_embed_layer=nn.Linear(embed_dim * 2, embed_dim),
            )
        )

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    elif method == 'mamba':

        features_embed_layer = FeaturesEmbedLayer(
            input_size=input_size,
            embed_dim=embed_dim,
            input_channel=1,
            depth=depth,
        )

        cashes_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim // 2,
        )

        positions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=num_max_tokens,
            embed_dim=embed_dim // 2,
        )

        actions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=action_dim,
            embed_dim=embed_dim * 2,
        )

        rets_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim * 2,
        )

        total_num_block_tokens = num_block_tokens * 3  # features + cashes + positions + actions + rets
        indices_embedding_layer = DiscreteEmbeddingLayer(
            num_max_tokens=total_num_block_tokens,
            embed_dim=embed_dim * 2,
        )

        encoder_layer = nn.ModuleDict(
            dict(
                features=features_embed_layer,
                cashes=cashes_embed_layer,
                positions=positions_embed_layer,  # position of the user in trading
                actions=actions_embed_layer,
                rets=rets_embed_layer,
                indices=indices_embedding_layer,  # position embedding of the sequence
            )
        )

        blocks = Mamba(
            d_model = embed_dim * 2,
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        norm = norm_layer(embed_dim * 2)

        middle_layer = nn.ModuleDict(
            dict(
                reduce_embed_layer=nn.Linear(embed_dim * 2, embed_dim),
            )
        )

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    elif method == 'mamba2':

        features_embed_layer = FeaturesEmbedLayer(
            input_size=input_size,
            embed_dim=embed_dim,
            input_channel=1,
            depth=depth,
        )

        cashes_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim // 2,
        )

        positions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=num_max_tokens,
            embed_dim=embed_dim // 2,
        )

        actions_embed_layer = DiscreteEmbeddingLayer(
            num_max_tokens=action_dim,
            embed_dim=embed_dim * 2,
        )

        rets_embed_layer = ContinuousEmbeddingLayer(
            embed_dim=embed_dim * 2,
        )

        total_num_block_tokens = num_block_tokens * 3  # features + cashes + positions + actions + rets
        indices_embedding_layer = DiscreteEmbeddingLayer(
            num_max_tokens=total_num_block_tokens,
            embed_dim=embed_dim * 2,
        )

        encoder_layer = nn.ModuleDict(
            dict(
                features=features_embed_layer,
                cashes=cashes_embed_layer,
                positions=positions_embed_layer,  # position of the user in trading
                actions=actions_embed_layer,
                rets=rets_embed_layer,
                indices=indices_embedding_layer,  # position embedding of the sequence
            )
        )

        blocks = Mamba2(
            d_model=embed_dim * 2,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        norm = norm_layer(embed_dim * 2)

        middle_layer = nn.ModuleDict(
            dict(
                reduce_embed_layer=nn.Linear(embed_dim * 2, embed_dim),
            )
        )

        decoder_layer = nn.Linear(
            embed_dim,
            output_dim,
            bias=True,
        )

    # Combine all layers into a single Sequential module
    modules = nn.ModuleDict({
        "encoder_layer": encoder_layer,
        "blocks": blocks,
        "norm": norm,
        "middle_layer": middle_layer,
        "decoder_layer": decoder_layer,
    })

    # Apply weight initialization
    modules.apply(init_weights)

    return modules