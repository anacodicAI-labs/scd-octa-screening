from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .encoders import EncoderSpec, TimmEncoder
from .fusion import FusionSpec, build_fusion


OptionName = Literal["optionA", "optionB", "optionC", "optionD"]


@dataclass(frozen=True)
class ModelSpec:
    option: OptionName
    encoder: EncoderSpec
    fusion: FusionSpec
    emb_dim: int = 256  # projection dim after encoder


class FourViewScreeningModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = TimmEncoder(spec.encoder, out_dim=spec.emb_dim)
        self.fusion = build_fusion(spec.fusion)
        self.head = nn.Linear(getattr(self.fusion, "out_dim"), 1)

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        x: Bx4x3xHxW
        view_mask: Bx4 (1 present, 0 missing)
        returns logits: B
        """
        b, v, c, h, w = x.shape
        x_flat = x.reshape(b * v, c, h, w)
        z_flat = self.encoder(x_flat)  # (B*4)xD
        z = z_flat.reshape(b, v, -1)  # Bx4xD
        u = self.fusion(z, view_mask)
        logits = self.head(u).squeeze(-1)
        return logits

