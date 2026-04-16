from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .encoders import EncoderSpec
from .fusion import FusionSpec
from .model import FourViewScreeningModel, ModelSpec


def spec_to_checkpoint_dict(spec: ModelSpec) -> dict[str, Any]:
    """Nested dicts only (stable across torch versions)."""
    return asdict(spec)


def model_spec_from_checkpoint_blob(spec_blob: Any) -> ModelSpec:
    """Rebuild ModelSpec from dataclass instance or nested dict (legacy checkpoints)."""
    if isinstance(spec_blob, ModelSpec):
        return spec_blob
    if not isinstance(spec_blob, dict):
        raise SystemExit("Checkpoint spec is not a dict or ModelSpec; re-train with current train.py.")

    enc_raw = spec_blob.get("encoder")
    if isinstance(enc_raw, EncoderSpec):
        enc = enc_raw
    elif isinstance(enc_raw, dict):
        enc = EncoderSpec(
            timm_name=str(enc_raw.get("timm_name", "efficientnet_b0")),
            pretrained=bool(enc_raw.get("pretrained", True)),
        )
    else:
        raise SystemExit("Checkpoint missing encoder spec.")

    fus_raw = spec_blob.get("fusion")
    if isinstance(fus_raw, FusionSpec):
        fusion = fus_raw
    elif isinstance(fus_raw, dict):
        fusion = FusionSpec(
            kind=fus_raw.get("kind", "concat_mlp"),
            emb_dim=int(fus_raw.get("emb_dim", 256)),
            mlp_hidden=int(fus_raw.get("mlp_hidden", 256)),
            dropout=float(fus_raw.get("dropout", 0.1)),
            attn_hidden=int(fus_raw.get("attn_hidden", 128)),
            n_heads=int(fus_raw.get("n_heads", 4)),
            n_layers=int(fus_raw.get("n_layers", 2)),
            ff_dim=int(fus_raw.get("ff_dim", 512)),
        )
    else:
        fusion = FusionSpec(kind="concat_mlp", emb_dim=int(spec_blob.get("emb_dim", 256)))

    return ModelSpec(
        option=spec_blob.get("option", "optionA"),  # type: ignore[arg-type]
        encoder=enc,
        fusion=fusion,
        emb_dim=int(spec_blob.get("emb_dim", 256)),
    )


def load_modern_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[FourViewScreeningModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = model_spec_from_checkpoint_blob(ckpt.get("spec"))
    model = FourViewScreeningModel(spec).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
