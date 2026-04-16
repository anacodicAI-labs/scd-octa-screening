from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

FusionKind = Literal["concat_mlp", "attn_pool", "tiny_transformer"]


@dataclass(frozen=True)
class FusionSpec:
    kind: FusionKind
    emb_dim: int

    # MLP head
    mlp_hidden: int = 256
    dropout: float = 0.1

    # Attention pooling
    attn_hidden: int = 128

    # Tiny transformer
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 512


class ConcatMLPFusion(nn.Module):
    def __init__(self, spec: FusionSpec) -> None:
        super().__init__()
        d = spec.emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, spec.mlp_hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.mlp_hidden, spec.mlp_hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.out_dim = int(spec.mlp_hidden)

    def forward(self, z: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        # z: Bx4xD, view_mask: Bx4 (1 present, 0 missing)
        z = z * view_mask.unsqueeze(-1)
        u = z.reshape(z.shape[0], -1)  # Bx(4D)
        return self.mlp(u)


class AttentionPoolingFusion(nn.Module):
    """
    Learn weights over the 4 views, then weighted-sum embeddings.
    """

    def __init__(self, spec: FusionSpec) -> None:
        super().__init__()
        d = spec.emb_dim
        self.score = nn.Sequential(
            nn.Linear(d, spec.attn_hidden),
            nn.Tanh(),
            nn.Linear(spec.attn_hidden, 1),
        )
        self.out_dim = d

    def forward(self, z: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        # z: Bx4xD
        scores = self.score(z).squeeze(-1)  # Bx4
        # Mask missing views by setting score to -inf so softmax weight -> 0
        neg_inf = torch.tensor(-1e9, device=z.device, dtype=z.dtype)
        scores = torch.where(view_mask > 0.5, scores, neg_inf)
        w = torch.softmax(scores, dim=1)  # Bx4
        u = (w.unsqueeze(-1) * z).sum(dim=1)  # BxD
        return u


class TinyTransformerFusion(nn.Module):
    """
    Treat the 4 view embeddings as 4 tokens and run a small TransformerEncoder.
    Output = masked mean pooling over tokens.
    """

    def __init__(self, spec: FusionSpec) -> None:
        super().__init__()
        d = spec.emb_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=spec.n_heads,
            dim_feedforward=spec.ff_dim,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=spec.n_layers)
        self.out_dim = d

    def forward(self, z: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask expects True for "pad" (missing), shape Bx4
        key_padding_mask = (view_mask <= 0.5)
        h = self.encoder(z, src_key_padding_mask=key_padding_mask)  # Bx4xD

        # masked mean pooling
        m = view_mask.unsqueeze(-1)  # Bx4x1
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        u = (h * m).sum(dim=1) / denom
        return u


def build_fusion(spec: FusionSpec) -> nn.Module:
    if spec.kind == "concat_mlp":
        return ConcatMLPFusion(spec)
    if spec.kind == "attn_pool":
        return AttentionPoolingFusion(spec)
    if spec.kind == "tiny_transformer":
        return TinyTransformerFusion(spec)
    raise ValueError(f"Unknown fusion kind: {spec.kind}")

