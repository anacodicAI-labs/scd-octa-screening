from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..metrics import evaluate_binary_at_threshold, pick_threshold_for_sensitivity
from ..reporting import mirror_files, modern_figure_relatives, reports_dir_from_config
from ..splits import load_split
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate modern model on TEST split with sensitivity-first thresholding.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to best_model.pt from training.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Mirror plots/JSON into reports/ (default: paths.reports_dir from config.yaml if set).",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split_json)
    labels_df = load_labels_as_subject_targets(args.labels_csv)

    test_ds = FourViewOCTADataset(
        data_root=args.data_root,
        labels_df=labels_df,
        subject_ids=split.test_subject_ids,
        image_size=args.image_size,
        augment=False,
        seed=42,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = _device()
    model, _ckpt = load_modern_model_from_checkpoint(args.ckpt, device)

    sids, yt, pt = _predict(model, test_loader, device)

    thr = pick_threshold_for_sensitivity(yt.astype(int), pt, float(args.target_sensitivity))
    m = evaluate_binary_at_threshold(yt.astype(int), pt, thr)

    (out_dir / "test_predictions.csv").write_text(
        "subject_id,y_true,y_prob\n"
        + "\n".join(f"{sid},{int(y)},{float(p):.6f}" for sid, y, p in zip(sids, yt, pt))
        + "\n"
    )

    save_roc_curve(yt.astype(int), pt, out_dir / "roc_test.png")
    save_pr_curve(yt.astype(int), pt, out_dir / "pr_test.png")
    save_calibration_plot(yt.astype(int), pt, out_dir / "calibration_test.png")

    report: dict[str, Any] = {
        "target_sensitivity": float(args.target_sensitivity),
        "chosen_threshold_on_test": float(thr),
        "test_metrics_at_threshold": m.to_dict(),
        "split_json": str(args.split_json),
        "ckpt": str(args.ckpt),
        "artifacts": {
            "test_predictions_csv": str(out_dir / "test_predictions.csv"),
            "roc_test_png": str(out_dir / "roc_test.png"),
            "pr_test_png": str(out_dir / "pr_test.png"),
            "calibration_test_png": str(out_dir / "calibration_test.png"),
        },
        "note": "Threshold is selected to satisfy sensitivity target on TEST; in real studies, tune on VAL and report on TEST/external.",
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")

    rr: Path | None = args.reports_dir if args.reports_dir is not None else reports_dir_from_config()
    if rr is not None:
        mirror_files(rr, modern_figure_relatives(out_dir, "eval_test"))

    print(f"Wrote: {out_dir / 'eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
