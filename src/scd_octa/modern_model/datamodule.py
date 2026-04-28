from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from ..io import (
    build_subject_records_nested_or_flat,
    merge_subject_records,
    subject_records_from_index_dataframe,
)
from ..splits import PatientSplit, load_split

ViewName = Literal["od_svp", "od_dcp", "os_svp", "os_dcp"]


def _read_tif_as_float01(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        # Some originals are RGB; we convert to grayscale by channel average.
        arr = arr.astype(np.float32).mean(axis=-1)
    arr = arr.astype(np.float32)
    # Robust-ish normalization for uint8 / uint16 etc.
    maxv = float(np.max(arr)) if arr.size else 0.0
    if maxv > 0:
        arr = arr / maxv
    return arr


def _to_3ch_tensor(img01: np.ndarray) -> torch.Tensor:
    # img01: HxW in [0,1]
    t = torch.from_numpy(img01).float().unsqueeze(0)  # 1xHxW
    t = t.repeat(3, 1, 1)  # 3xHxW for timm models
    return t


def cnn_extraction(path: Path, image_size: int = 224) -> torch.Tensor:
    """
    Read one OCTA TIFF image and return a 3xHxW tensor for CNN backbones.
    """
    img01 = _read_tif_as_float01(path)
    x = _to_3ch_tensor(img01)
    x = torch.nn.functional.interpolate(
        x.unsqueeze(0),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return x


@dataclass(frozen=True)
class LabelConfig:
    """
    We support both:
      - subject-level labels: subject_id,label
      - eye-level labels: subject_id,eye,octa_abnormal_label,gradable

    For this project, modern model uses SUBJECT-level target by default.
    If eye-level labels are provided, we reduce them to subject-level with:
      subject_label = max(label over gradable eyes)
    """

    label_col: str = "octa_abnormal_label"
    gradable_col: str = "gradable"
    eye_col: str = "eye"


def load_labels_as_subject_targets(labels_csv: Path, cfg: LabelConfig = LabelConfig()) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if "subject_id" not in df.columns:
        raise SystemExit("dataset_index.csv must contain subject_id.")
    df["subject_id"] = df["subject_id"].astype(str)

    # Subject-level labels (preferred when present): subject_id + label
    if "label" in df.columns:
        out = df[["subject_id", "label"]].copy()
        out["label"] = out["label"].astype(int)
        if "gradable" in df.columns:
            out["gradable"] = df["gradable"].astype(int)
        else:
            out["gradable"] = 1
        return out

    # Eye-level format: octa_abnormal_label + gradable (+ optional eye)
    needed = {cfg.label_col, cfg.gradable_col}
    missing = needed.difference(df.columns)
    if missing:
        raise SystemExit(
            "dataset_index.csv needs either:\n"
            "  - subject-level: columns subject_id, label (0/1), optional gradable\n"
            "  - eye-level: columns subject_id, octa_abnormal_label, gradable (and usually eye)\n"
            f"Missing for eye-level mode: {sorted(missing)}"
        )

    df[cfg.gradable_col] = df[cfg.gradable_col].astype(int)
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    # If eye column exists, reduce per subject. If not, assume already subject-level.
    if cfg.eye_col in df.columns:
        gradable_df = df[df[cfg.gradable_col] == 1]
        if len(gradable_df) == 0:
            raise SystemExit("No gradable rows found in dataset_index.csv (gradable==1).")
        agg = (
            gradable_df.groupby("subject_id")[cfg.label_col]
            .max()
            .rename("label")
            .reset_index()
        )
        agg["gradable"] = 1
        return agg

    out = df[["subject_id", cfg.label_col]].rename(columns={cfg.label_col: "label"}).copy()
    out["label"] = out["label"].astype(int)
    out["gradable"] = df[cfg.gradable_col].astype(int) if cfg.gradable_col in df.columns else 1
    return out


def build_merged_record_map(data_root: Path, index_csv: Path | None) -> dict[str, SubjectRecord]:
    """Filesystem under ``data_root`` plus optional path columns from ``index_csv`` (e.g. dataset_index.csv)."""
    fs_records = build_subject_records_nested_or_flat(data_root)
    fs_by_id = {r.subject_id: r for r in fs_records}
    csv_by_id: dict[str, SubjectRecord] = {}
    if index_csv is not None and index_csv.is_file():
        full_df = pd.read_csv(index_csv)
        csv_by_id = subject_records_from_index_dataframe(full_df)
    all_ids = set(fs_by_id) | set(csv_by_id)
    out: dict[str, SubjectRecord] = {}
    for sid in all_ids:
        merged = merge_subject_records(fs_by_id.get(sid), csv_by_id.get(sid))
        if merged is not None:
            out[sid] = merged
    return out


class FourViewOCTADataset(Dataset):
    def __init__(
        self,
        *,
        data_root: Path,
        labels_df: pd.DataFrame,
        subject_ids: list[str],
        image_size: int = 224,
        augment: bool = False,
        seed: int = 42,
        index_csv: Path | None = None,
    ) -> None:
        self.data_root = data_root
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(int(seed))

        self.record_by_id = build_merged_record_map(data_root, index_csv)

        labels_df = labels_df.copy()
        labels_df["subject_id"] = labels_df["subject_id"].astype(str)
        labels_df = labels_df.set_index("subject_id")

        self.items: list[tuple[str, int]] = []
        for sid in subject_ids:
            if sid not in self.record_by_id:
                continue
            if sid not in labels_df.index:
                continue
            y = int(labels_df.loc[sid, "label"])
            self.items.append((sid, y))

        if len(self.items) == 0:
            raise SystemExit("No training items found after joining labels with available images.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_view(self, sid: str, view: ViewName) -> tuple[torch.Tensor, int]:
        r = self.record_by_id[sid]
        p: Optional[Path] = getattr(r, view)
        if p is None:
            # Missing view: return zeros and mark mask=0
            x = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            return x, 0

        x = cnn_extraction(p, image_size=self.image_size)
        return x, 1

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: 3xHxW
        if not self.augment:
            return x
        # Simple, realistic augmentations for OCTA en-face:
        # flips + mild brightness/contrast jitter.
        if self.rng.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if self.rng.random() < 0.2:
            x = torch.flip(x, dims=[1])  # vertical (less common but ok)
        if self.rng.random() < 0.5:
            b = float(self.rng.uniform(-0.05, 0.05))
            c = float(self.rng.uniform(0.9, 1.1))
            x = torch.clamp((x + b) * c, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int) -> dict:
        sid, y = self.items[idx]
        views: list[ViewName] = ["od_svp", "od_dcp", "os_svp", "os_dcp"]

        xs: list[torch.Tensor] = []
        masks: list[int] = []
        for v in views:
            x, m = self._load_view(sid, v)
            x = self._augment(x)
            xs.append(x)
            masks.append(m)

        X = torch.stack(xs, dim=0)  # 4x3xHxW
        view_mask = torch.tensor(masks, dtype=torch.float32)  # 4
        y_t = torch.tensor(float(y), dtype=torch.float32)

        return {
            "subject_id": sid,
            "x": X,
            "view_mask": view_mask,
            "y": y_t,
        }


def load_split_from_path(path: Path) -> PatientSplit:
    return load_split(path)


def save_subject_id_manifest(subject_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"subject_ids": list(subject_ids)}, indent=2) + "\n")

