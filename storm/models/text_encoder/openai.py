import torch
from torch import nn
import numpy as np
from typing import Optional, List

from storm.registry import MODEL
from storm.provider import OpenAIProvider
from storm.utils import load_json
from storm.utils import assemble_project_path

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

@MODEL.register_module(force=True)
class OpenAITextEncoder(nn.Module):
    def __init__(self,
                 *args,
                 provider_cfg_path: str,
                 if_reduce_dim: bool = True,
                 reduced_dim: int = 1024,
                 **kwargs):
        super(OpenAITextEncoder, self).__init__()
        self.provider_cfg_path = assemble_project_path(provider_cfg_path)
        self.if_reduce_dim = if_reduce_dim
        self.reduced_dim = reduced_dim

        provider_cfg = load_json(self.provider_cfg_path)
        self.pretrained_model = provider_cfg["emb_model"]

        self.provider = OpenAIProvider(self.provider_cfg_path)

        self.embed_dim = self.provider.get_embedding_dim() if not self.if_reduce_dim else self.reduced_dim
        self.model_max_length = self.provider.get_max_tokens(self.pretrained_model)

    def __str__(self):
        str = f"OpenAITextEncoder from pretrained model: {self.pretrained_model}.\n"
        str += f"Embedding dimension: {self.embed_dim}.\n"
        str += f"Model max length: {self.model_max_length}.\n"
        return str

    def encode(self, text: Optional[str|List[str]] = None) -> torch.Tensor:
        embedding = self.provider.embed_documents(text)
        embedding = np.array(embedding)
        if self.if_reduce_dim:
            reduced_dim = min(self.reduced_dim, self.embed_dim)
            embedding = embedding[:, :reduced_dim]
            embedding = normalize_l2(embedding)
        embedding = torch.tensor(embedding)
        return embedding

    def forward(self, text: Optional[str|List[str]] = None) -> torch.Tensor:
        return self.encode(text)