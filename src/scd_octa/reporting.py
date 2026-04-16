from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import yaml


def reports_dir_from_config(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return `paths.reports_dir` from config.yaml if present, else None."""
    p = Path("config.yaml") if config_path is None else config_path
    if not p.is_file():
        return None
    data = yaml.safe_load(p.read_text()) or {}
    rd = (data.get("paths") or {}).get("reports_dir")
    if not rd:
        return None
    return Path(str(rd))


def mirror_files(reports_root: Path, pairs: list[tuple[Path, Path]]) -> None:
    """
    Copy each (src, rel_under_reports_root) if src exists.
    Creates parent directories for rel.
    """
    for src, rel in pairs:
        if not src.is_file():
            continue
        dest = reports_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def modern_figure_relatives(out_dir: Path, phase: str) -> list[tuple[Path, Path]]:
    """
    phase: 'train_val' | 'eval_test'
    out_dir: e.g. results/modern/optionA
    """
    name = out_dir.name
    base = Path("figures") / "modern" / name
    if phase == "train_val":
        return [
            (out_dir / "roc_val.png", base / "roc_val.png"),
            (out_dir / "pr_val.png", base / "pr_val.png"),
            (out_dir / "calibration_val.png", base / "calibration_val.png"),
            (out_dir / "train_report.json", Path("tables") / "modern" / name / "train_report.json"),
        ]
    return [
        (out_dir / "roc_test.png", base / "roc_test.png"),
        (out_dir / "pr_test.png", base / "pr_test.png"),
        (out_dir / "calibration_test.png", base / "calibration_test.png"),
        (out_dir / "eval_report.json", Path("tables") / "modern" / name / "eval_report.json"),
    ]


def baseline_figure_relatives(out_dir: Path) -> list[tuple[Path, Path]]:
    """out_dir: results/baseline"""
    base = Path("figures") / "baseline"
    return [
        (out_dir / "roc_val.png", base / "roc_val.png"),
        (out_dir / "roc_test.png", base / "roc_test.png"),
        (out_dir / "pr_val.png", base / "pr_val.png"),
        (out_dir / "pr_test.png", base / "pr_test.png"),
        (out_dir / "calibration_val.png", base / "calibration_val.png"),
        (out_dir / "calibration_test.png", base / "calibration_test.png"),
        (out_dir / "train_eval_report.json", Path("tables") / "baseline" / "train_eval_report.json"),
    ]
