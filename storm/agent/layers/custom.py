import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from timm.models.layers import to_2tuple
from torch.nn.functional import softmax

class FeaturesEmbedLayer(nn.Module):
    def __init__(self,
                 input_size = (4, 152),
                 input_channel: int = 1,
                 depth: int = 2,
                 embed_dim: int = 128,
                 ):
        super(FeaturesEmbedLayer, self).__init__()

        self.input_size = to_2tuple(input_size)
        self.embed_dim = embed_dim
        self.depth = depth

        self.proj = nn.Conv2d(in_channels=input_channel, out_channels = embed_dim, kernel_size=self.input_size, stride=self.input_size)

        self.initialize_weights()


    def initialize_weights(self):

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):

        B, N, D, F = x.shape
        x = x.unsqueeze(2)

        x = rearrange(x, 'b n c d f -> (b n) c d f', b = B, n = N)
        x = self.proj(x).squeeze(-1).squeeze(-1)

        x = rearrange(x, '(b n) c -> b n c', b=B, n=N)

        return x

class ContinuousEmbeddingLayer(nn.Module):

    def __init__(self,
                 embed_dim: int = 256,
                 input_dim = 1
                 ):
        super(ContinuousEmbeddingLayer, self).__init__()

        self.proj = nn.Linear(input_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
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
    def forward(self, x):

        if len(x.shape) <= 3:
            x = x.unsqueeze(-1)

        x = self.proj(x)

        return x

class DiscreteEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_max_tokens = 1000,
                 embed_dim: int = 256
                 ):
        super().__init__()

        self.embed_layer = nn.Embedding(num_max_tokens, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):
        x = self.embed_layer(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(state_dim, embed_dim)
        self.key = nn.Linear(action_dim, embed_dim)
        self.value = nn.Linear(action_dim, embed_dim)

    def forward(self, state, action):
        """
        Adapts to input shapes:
        - 2D input: (batch_size, feature_dim)
        - 3D input: (batch_size, num_envs, feature_dim)
        """
        # Check if the input is 3D
        is_3d = state.dim() == 3 and action.dim() == 3

        # If 2D input, add an environment dimension
        if not is_3d:
            state = state.unsqueeze(1)  # (batch_size, 1, feature_dim)
            action = action.unsqueeze(1)  # (batch_size, 1, feature_dim)

        # Generate Query, Key, and Value
        q = self.query(state)  # (batch_size, num_envs, embed_dim)
        k = self.key(action)  # (batch_size, num_envs, embed_dim)
        v = self.value(action)  # (batch_size, num_envs, embed_dim)

        # Compute attention weights
        attention_weights = softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)  # (batch_size, num_envs, num_envs)

        # Apply attention to Value
        fused = torch.bmm(attention_weights, v)  # (batch_size, num_envs, embed_dim)

        # If input was 2D, remove the added environment dimension
        if not is_3d:
            fused = fused.squeeze(1)  # (batch_size, embed_dim)

        return fused
