from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EncoderSpec:
    """
    We use `timm` model names so we can swap CNN/ViT backbones easily.
    Examples:
      - efficientnet_b0
      - densenet121
      - convnext_tiny
      - swin_tiny_patch4_window7_224
    """

    timm_name: str
    pretrained: bool = True


class TimmEncoder(nn.Module):
    def __init__(self, spec: EncoderSpec, *, out_dim: Optional[int] = None) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: timm. From the repo root: pip install -r requirements.txt "
                "(or use the project's .venv — `make` picks .venv/bin/python when present)."
            ) from e

        self.backbone = timm.create_model(
            spec.timm_name,
            pretrained=bool(spec.pretrained),
            num_classes=0,  # timm: return pooled features
            global_pool="avg",
        )

        # Infer feature dim
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone(dummy)
        feat_dim = int(feat.shape[-1])

        self.out_dim = int(out_dim) if out_dim is not None else feat_dim
        self.proj = nn.Identity() if self.out_dim == feat_dim else nn.Linear(feat_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)  # BxD
        z = self.proj(z)
        return z

