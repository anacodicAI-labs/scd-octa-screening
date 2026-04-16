"""
Evaluate a trained modern checkpoint on an **external** cohort (different data root / labels).

All subjects present in labels.csv (after the same gradable + join rules as training) who
also have image files under ``data_root`` are scored — no split JSON required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..io import build_subject_records
from ..metrics import evaluate_binary_at_threshold, pick_threshold_for_sensitivity
from ..reporting import mirror_files, modern_figure_relatives, reports_dir_from_config
from ..viz import save_calibration_plot, save_pr_curve, save_roc_curve
from .ckpt import load_modern_model_from_checkpoint
from .datamodule import FourViewOCTADataset, load_labels_as_subject_targets


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    sids: list[str] = []
    ys: list[float] = []
    ps: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        vm = batch["view_mask"].to(device)
        y = batch["y"].cpu().numpy().astype(float)
        logits = model(x, vm).detach().cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        sids.extend([str(s) for s in batch["subject_id"]])
        ys.extend(y.tolist())
        ps.extend(prob.tolist())
    return sids, np.array(ys, dtype=float), np.array(ps, dtype=float)


def _external_subject_ids(labels_df: Any, data_root: Path) -> list[str]:
    records = build_subject_records(data_root)
    have = {r.subject_id for r in records}
    ids = [str(s) for s in labels_df["subject_id"].astype(str).tolist()]
    # preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for sid in ids:
        if sid in have and sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a frozen checkpoint on an external OCTA cohort.")
    ap.add_argument("--external-data-root", type=Path, required=True, help="Root folder of external OCTA exports.")
    ap.add_argument("--external-labels-csv", type=Path, required=True, help="Labels CSV (same columns as training labels.csv).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="Write predictions + plots + eval_report.json here.")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Mirror plots into reports/ (default: paths.reports_dir from config.yaml if set).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels_as_subject_targets(args.external_labels_csv)
    subject_ids = _external_subject_ids(labels_df, args.external_data_root)
    if len(subject_ids) == 0:
        raise SystemExit("No subjects overlap external labels and images under external-data-root.")

    ds = FourViewOCTADataset(
        data_root=args.external_data_root,
        labels_df=labels_df,
        subject_ids=subject_ids,
        image_size=args.image_size,
        augment=False,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = _device()
    model, _ckpt = load_modern_model_from_checkpoint(args.ckpt, device)
    sids, yt, pt = _predict(model, loader, device)

    thr = pick_threshold_for_sensitivity(yt.astype(int), pt, float(args.target_sensitivity))
    m = evaluate_binary_at_threshold(yt.astype(int), pt, thr)

    (out_dir / "external_predictions.csv").write_text(
        "subject_id,y_true,y_prob\n"
        + "\n".join(f"{sid},{int(y)},{float(p):.6f}" for sid, y, p in zip(sids, yt, pt))
        + "\n"
    )

    if len(np.unique(yt.astype(int))) > 1:
        save_roc_curve(yt.astype(int), pt, out_dir / "roc_external.png")
        save_pr_curve(yt.astype(int), pt, out_dir / "pr_external.png")
        save_calibration_plot(yt.astype(int), pt, out_dir / "calibration_external.png")

    report: dict[str, Any] = {
        "cohort": "external",
        "external_data_root": str(args.external_data_root),
        "external_labels_csv": str(args.external_labels_csv),
        "n_subjects_scored": int(len(sids)),
        "target_sensitivity": float(args.target_sensitivity),
        "chosen_threshold_on_external": float(thr),
        "metrics_at_threshold": m.to_dict(),
        "ckpt": str(args.ckpt),
        "note": (
            "External run: threshold is tuned on this external set for a comparable sensitivity readout. "
            "For unbiased external performance, fix τ from internal VAL and only report metrics here."
        ),
    }
    (out_dir / "eval_external_report.json").write_text(json.dumps(report, indent=2) + "\n")

    rr: Path | None = args.reports_dir if args.reports_dir is not None else reports_dir_from_config()
    if rr is not None:
        name = out_dir.name
        base = Path("figures") / "modern_external" / name
        pairs = [
            (out_dir / "roc_external.png", base / "roc_external.png"),
            (out_dir / "pr_external.png", base / "pr_external.png"),
            (out_dir / "calibration_external.png", base / "calibration_external.png"),
            (out_dir / "eval_external_report.json", Path("tables") / "modern_external" / name / "eval_external_report.json"),
        ]
        mirror_files(rr, pairs)

    print(f"Wrote: {out_dir / 'eval_external_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
