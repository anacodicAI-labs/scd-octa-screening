from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .io import build_subject_records


def vessel_density_from_mask(mask: np.ndarray) -> float:
    """
    Very simple proxy:
    - assume vessels are brighter (often 255) and background is darker (0)
    - vessel density = fraction of pixels > 0

    This is NOT a clinical-grade definition. It's a first-pass feature for class demos.
    """
    if mask.ndim != 2:
        # If the mask somehow comes as (H,W,1) or (H,W,3), collapse safely.
        mask = mask[..., 0]
    total = mask.size
    if total == 0:
        return float("nan")
    return float((mask > 0).sum() / total)


def read_tif_gray(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    # If RGB, convert to grayscale by taking one channel (binarized should already be 2D)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute simple vessel-density features from binarized OCTA masks.")
    ap.add_argument("--data-root", type=Path, required=True, help="Path to scd-data directory.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_subject_records(args.data_root)

    rows: list[dict[str, object]] = []
    for r in records:
        row: dict[str, object] = {"subject_id": r.subject_id}

        for name, p in [
            ("od_svp", r.od_svp_bin),
            ("od_dcp", r.od_dcp_bin),
            ("os_svp", r.os_svp_bin),
            ("os_dcp", r.os_dcp_bin),
        ]:
            if p is None:
                row[f"{name}_vessel_density"] = np.nan
                row[f"{name}_mask_path"] = ""
                continue

            try:
                mask = read_tif_gray(p)
                row[f"{name}_vessel_density"] = vessel_density_from_mask(mask)
                row[f"{name}_mask_path"] = str(p)
                row[f"{name}_mask_h"] = int(mask.shape[0])
                row[f"{name}_mask_w"] = int(mask.shape[1])
            except Exception as e:
                row[f"{name}_vessel_density"] = np.nan
                row[f"{name}_mask_path"] = str(p)
                row[f"{name}_error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "biomarkers_vessel_density.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

