from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..clinical_tasks import require_screening_task
from ..metrics import evaluate_binary_at_threshold, pick_threshold_for_sensitivity
from ..reporting import mirror_files, modern_figure_relatives, reports_dir_from_config
from ..splits import load_split
from ..viz import save_calibration_plot, save_pr_curve, save_roc_curve
from .ckpt import spec_to_checkpoint_dict
from .datamodule import FourViewOCTADataset, load_labels_as_subject_targets
from .encoders import EncoderSpec
from .fusion import FusionSpec
from .model import FourViewScreeningModel, ModelSpec


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _spec_for_option(option: str, *, fusion_kind: str, encoder_timm: str | None) -> ModelSpec:
    option = option.strip()
    if option == "optionA":
        enc = EncoderSpec("efficientnet_b0", pretrained=True)
        fusion = FusionSpec(kind=fusion_kind, emb_dim=256)
        spec = ModelSpec(option="optionA", encoder=enc, fusion=fusion, emb_dim=256)
    elif option == "optionB":
        enc = EncoderSpec("convnext_tiny", pretrained=True)
        fusion = FusionSpec(kind=fusion_kind, emb_dim=256)
        spec = ModelSpec(option="optionB", encoder=enc, fusion=fusion, emb_dim=256)
    elif option == "optionC":
        enc = EncoderSpec("swin_tiny_patch4_window7_224", pretrained=True)
        fusion = FusionSpec(kind=fusion_kind, emb_dim=256)
        spec = ModelSpec(option="optionC", encoder=enc, fusion=fusion, emb_dim=256)
    elif option == "optionD":
        enc = EncoderSpec("convnext_tiny", pretrained=True)
        fusion = FusionSpec(kind="tiny_transformer", emb_dim=256, n_heads=4, n_layers=2)
        spec = ModelSpec(option="optionD", encoder=enc, fusion=fusion, emb_dim=256)
    else:
        raise SystemExit("option must be one of: optionA, optionB, optionC, optionD")

    if encoder_timm:
        enc_ov = EncoderSpec(encoder_timm.strip(), pretrained=True)
        spec = ModelSpec(option=spec.option, encoder=enc_ov, fusion=spec.fusion, emb_dim=spec.emb_dim)
    return spec


@torch.no_grad()
def _predict(model: FourViewScreeningModel, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
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
    ap = argparse.ArgumentParser(description="Train modern 4-view OCTA model (Options A–D).")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--option", type=str, required=True, help="optionA|optionB|optionC|optionD")
    ap.add_argument(
        "--fusion",
        type=str,
        default="concat_mlp",
        help="concat_mlp|attn_pool (ignored for optionD which uses tiny_transformer)",
    )
    ap.add_argument(
        "--encoder-timm",
        type=str,
        default=None,
        help="Override timm backbone name for all options, e.g. efficientnet_b2, densenet121, swinv2_tiny_window8_256.",
    )
    ap.add_argument(
        "--clinical-task",
        type=str,
        default="screening",
        help="Only 'screening' is implemented (binary abnormal vs normal).",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Mirror plots/JSON into reports/ (default: paths.reports_dir from config.yaml if set).",
    )
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    args = ap.parse_args()

    require_screening_task(str(args.clinical_task))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split_json)
    labels_df = load_labels_as_subject_targets(args.labels_csv)

    train_ds = FourViewOCTADataset(
        data_root=args.data_root,
        labels_df=labels_df,
        subject_ids=split.train_subject_ids,
        image_size=args.image_size,
        augment=True,
        seed=args.seed,
        index_csv=args.labels_csv,
    )
    val_ds = FourViewOCTADataset(
        data_root=args.data_root,
        labels_df=labels_df,
        subject_ids=split.val_subject_ids,
        image_size=args.image_size,
        augment=False,
        seed=args.seed,
        index_csv=args.labels_csv,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    enc_arg = str(args.encoder_timm).strip() if args.encoder_timm else None
    spec = _spec_for_option(args.option, fusion_kind=args.fusion, encoder_timm=enc_arg or None)
    model = FourViewScreeningModel(spec)
    device = _device()
    model.to(device)

    # class imbalance: pos_weight = n_neg/n_pos (train split)
    y_train = np.array([int(y) for _, y in train_ds.items], dtype=int)  # type: ignore[attr-defined]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(1, n_pos))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_auc = -1.0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            vm = batch["view_mask"].to(device)
            y = batch["y"].to(device)
            logits = model(x, vm)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        # quick val AUC and checkpointing
        _, yv, pv = _predict(model, val_loader, device)
        from ..metrics import safe_auc

        val_auc = safe_auc(yv.astype(int), pv)
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            torch.save(
                {"model_state": model.state_dict(), "spec": spec_to_checkpoint_dict(spec), "epoch": epoch},
                best_path,
            )

    if not best_path.is_file():
        # Val AUC can be undefined (single-class val split); still persist last weights for eval.
        torch.save(
            {"model_state": model.state_dict(), "spec": spec_to_checkpoint_dict(spec), "epoch": int(args.epochs)},
            best_path,
        )

    # Load best and write a small training report on VAL (for tuning/visibility)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    sids_v, yv, pv = _predict(model, val_loader, device)
    thr = pick_threshold_for_sensitivity(yv.astype(int), pv, float(args.target_sensitivity))
    m = evaluate_binary_at_threshold(yv.astype(int), pv, thr)

    (out_dir / "val_predictions.csv").write_text(
        "subject_id,y_true,y_prob\n"
        + "\n".join(f"{sid},{int(y)},{float(p):.6f}" for sid, y, p in zip(sids_v, yv, pv))
        + "\n"
    )

    # Plots on val
    save_roc_curve(yv.astype(int), pv, out_dir / "roc_val.png")
    save_pr_curve(yv.astype(int), pv, out_dir / "pr_val.png")
    save_calibration_plot(yv.astype(int), pv, out_dir / "calibration_val.png")

    report: dict[str, Any] = {
        "option": str(args.option),
        "fusion": str(spec.fusion.kind),
        "encoder": spec.encoder.timm_name,
        "emb_dim": int(spec.emb_dim),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight": float(pos_weight),
        },
        "split_json": str(args.split_json),
        "val_metrics_at_threshold": m.to_dict(),
        "target_sensitivity": float(args.target_sensitivity),
        "chosen_threshold_on_val": float(thr),
        "artifacts": {
            "best_model_pt": str(best_path),
            "val_predictions_csv": str(out_dir / "val_predictions.csv"),
            "roc_val_png": str(out_dir / "roc_val.png"),
            "pr_val_png": str(out_dir / "pr_val.png"),
            "calibration_val_png": str(out_dir / "calibration_val.png"),
        },
        "note": (
            "This report is computed on VAL for visibility. Final performance must be computed on TEST "
            "using modern_model/eval.py without re-tuning on test."
        ),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")

    rr: Path | None = args.reports_dir if args.reports_dir is not None else reports_dir_from_config()
    if rr is not None:
        mirror_files(rr, modern_figure_relatives(out_dir, "train_val"))

    print(f"Wrote: {out_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

