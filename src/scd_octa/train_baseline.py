from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from .modern_model.datamodule import load_labels_as_subject_targets
from .reporting import baseline_figure_relatives, mirror_files, reports_dir_from_config
from .splits import load_split
from .viz import save_calibration_plot, save_pr_curve, save_roc_curve

ModelType = Literal["logistic", "random_forest"]


@dataclass(frozen=True)
class EvalResult:
    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    n_test: int


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _specificity_from_cm(cm: np.ndarray) -> float:
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def _pick_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> float:
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.concatenate([thresholds, np.array([0.0, 1.0])])
    thresholds = np.unique(thresholds)
    thresholds.sort()

    best = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        if sens >= target_sensitivity:
            best = float(t)
    return best


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalResult:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = _specificity_from_cm(cm)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = _safe_auc(y_true, y_prob)

    return EvalResult(
        threshold=float(threshold),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1=float(f1),
        accuracy=float(acc),
        auc=float(auc),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        n_test=int(len(y_true)),
    )


def _build_sklearn_model(model_type: ModelType, seed: int) -> LogisticRegression | RandomForestClassifier:
    if model_type == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced_subsamples",
            random_state=seed,
            n_jobs=-1,
        )
    raise SystemExit("model-type must be logistic or random_forest")


def _maybe_plot_split(
    y_val: np.ndarray | None,
    p_val: np.ndarray | None,
    y_test: np.ndarray | None,
    p_test: np.ndarray | None,
    out_dir: Path,
) -> None:
    if y_val is not None and p_val is not None and len(y_val) > 0 and len(np.unique(y_val)) > 1:
        save_roc_curve(y_val.astype(int), p_val, out_dir / "roc_val.png")
        save_pr_curve(y_val.astype(int), p_val, out_dir / "pr_val.png")
        save_calibration_plot(y_val.astype(int), p_val, out_dir / "calibration_val.png")
    if y_test is not None and p_test is not None and len(y_test) > 0 and len(np.unique(y_test)) > 1:
        save_roc_curve(y_test.astype(int), p_test, out_dir / "roc_test.png")
        save_pr_curve(y_test.astype(int), p_test, out_dir / "pr_test.png")
        save_calibration_plot(y_test.astype(int), p_test, out_dir / "calibration_test.png")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train baseline (logistic regression or random forest) on biomarker features; "
        "sensitivity-first threshold. Use --split-json to match modern_model splits."
    )
    ap.add_argument("--biomarkers-csv", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Baseline results directory (e.g. results/baseline).",
    )
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Sklearn classifier for biomarker baseline.",
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Patient split JSON (same as modern training). Train/val/test rows chosen by subject_id.",
    )
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--test-size", type=float, default=0.3, help="Used only when --split-json is omitted.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="If set, copy plots/JSON into reports/ tree (defaults: read paths.reports_dir from config.yaml).",
    )
    args = ap.parse_args()

    model_type: ModelType = args.model_type.strip().lower()  # type: ignore[assignment]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"baseline_{model_type}.joblib"

    labels_df = load_labels_as_subject_targets(args.labels_csv)
    if "gradable" in labels_df.columns:
        labels_df = labels_df[labels_df["gradable"].astype(int) == 1].copy()
    if labels_df.empty:
        raise SystemExit("No gradable labeled subjects after loading labels.csv.")

    bio = pd.read_csv(args.biomarkers_csv)
    df = bio.merge(labels_df[["subject_id", "label"]], on="subject_id", how="inner")
    df = df.dropna(subset=["label"])
    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(int)

    feature_cols = [
        "od_svp_vessel_density",
        "od_dcp_vessel_density",
        "os_svp_vessel_density",
        "os_dcp_vessel_density",
    ]
    for c in feature_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing biomarker column: {c}")

    use_split = args.split_json is not None and args.split_json.is_file()

    # --- train-only tiny demo (no split file, very few subjects) ---
    train_only = (not use_split) and len(df) < 4
    if train_only:
        model = _build_sklearn_model(model_type, args.seed)
        X = df[feature_cols].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        model.fit(X, y)
        joblib.dump(model, model_path)
        report: dict[str, Any] = {
            "mode": "train_only",
            "model_type": model_type,
            "feature_cols": feature_cols,
            "n_labeled_subjects_used": int(len(df)),
            "artifacts": {"model_path": str(model_path)},
            "note": (
                "Fewer than 4 labeled subjects and no split JSON: no train/test evaluation. "
                "Add labels and run `make split` + `make train` with --split-json for metrics."
            ),
        }
        if model_type == "logistic":
            report["model"] = {
                "type": str(model),
                "coef": dict(zip(feature_cols, model.coef_.reshape(-1).tolist())),
                "intercept": float(model.intercept_[0]),
            }
        else:
            report["model"] = {
                "type": str(model),
                "feature_importances": dict(zip(feature_cols, model.feature_importances_.tolist())),
            }
        (out_dir / "train_eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote: {out_dir / 'train_eval_report.json'}")
        print(f"Wrote: {model_path}")
        return 0

    # --- split-json path: align with modern_model ---
    if use_split:
        split = load_split(args.split_json)
        train_ids = set(str(x) for x in split.train_subject_ids)
        val_ids = set(str(x) for x in split.val_subject_ids)
        test_ids = set(str(x) for x in split.test_subject_ids)

        train_df = df[df["subject_id"].isin(train_ids)].copy()
        val_df = df[df["subject_id"].isin(val_ids)].copy()
        test_df = df[df["subject_id"].isin(test_ids)].copy()

        if len(train_df) == 0:
            raise SystemExit("No labeled subjects overlap split train_subject_ids and biomarkers.")
        if len(np.unique(train_df["label"].to_numpy())) < 2:
            raise SystemExit("Train split needs both classes (0 and 1) for this baseline.")

        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=int)
        model = _build_sklearn_model(model_type, args.seed)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

        p_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if len(val_df) > 0:
            p_val = model.predict_proba(val_df[feature_cols].to_numpy(dtype=float))[:, 1]
            y_val = val_df["label"].to_numpy(dtype=int)

        p_test: np.ndarray | None = None
        y_test: np.ndarray | None = None
        if len(test_df) > 0:
            p_test = model.predict_proba(test_df[feature_cols].to_numpy(dtype=float))[:, 1]
            y_test = test_df["label"].to_numpy(dtype=int)

        # Threshold: prefer VAL; else TRAIN (tiny cohorts)
        if y_val is not None and len(y_val) > 0 and len(np.unique(y_val)) >= 2:
            thr_source = "val"
            threshold = _pick_threshold_for_sensitivity(y_val, p_val, float(args.target_sensitivity))
        else:
            thr_source = "train"
            p_tr = model.predict_proba(X_train)[:, 1]
            threshold = _pick_threshold_for_sensitivity(y_train, p_tr, float(args.target_sensitivity))

        if y_test is not None and len(y_test) > 0:
            result = evaluate_at_threshold(y_test, p_test, threshold)
            eval_split = "test"
        elif y_val is not None and len(y_val) > 0:
            result = evaluate_at_threshold(y_val, p_val, threshold)
            eval_split = "val"
        else:
            p_tr = model.predict_proba(X_train)[:, 1]
            result = evaluate_at_threshold(y_train, p_tr, threshold)
            eval_split = "train"

        _maybe_plot_split(y_val, p_val, y_test, p_test, out_dir)

        report = {
            "mode": "split_json",
            "model_type": model_type,
            "split_json": str(args.split_json),
            "threshold_tuned_on": thr_source,
            "metrics_reported_on": eval_split,
            "target_sensitivity": float(args.target_sensitivity),
            "chosen_threshold": float(threshold),
            "feature_cols": feature_cols,
            "artifacts": {"model_path": str(model_path)},
            "counts": {
                "train_subjects": int(len(train_df)),
                "val_subjects": int(len(val_df)),
                "test_subjects": int(len(test_df)),
            },
            "metrics_at_threshold": result.__dict__,
            "note": (
                "Threshold is chosen on VAL when available (else TRAIN). "
                "Final row reports metrics on TEST when available (else VAL, else TRAIN)."
            ),
        }
        if model_type == "logistic":
            report["model"] = {
                "type": "LogisticRegression",
                "coef": dict(zip(feature_cols, model.coef_.reshape(-1).tolist())),
                "intercept": float(model.intercept_[0]),
            }
        else:
            report["model"] = {
                "type": "RandomForestClassifier",
                "feature_importances": dict(zip(feature_cols, model.feature_importances_.tolist())),
            }

        (out_dir / "train_eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if y_test is not None and len(y_test) > 0:
            test_df.assign(pred_prob=p_test).to_csv(out_dir / "test_predictions.csv", index=False)
        elif y_val is not None and len(y_val) > 0:
            val_df.assign(pred_prob=p_val).to_csv(out_dir / "test_predictions.csv", index=False)

        rr = args.reports_dir if args.reports_dir is not None else reports_dir_from_config()
        if rr is not None:
            mirror_files(rr, baseline_figure_relatives(out_dir))

        print(f"Wrote: {out_dir / 'train_eval_report.json'}")
        print(f"Wrote: {model_path}")
        return 0

    # --- legacy: random GroupShuffleSplit (no split JSON) ---
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["subject_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = _build_sklearn_model(model_type, args.seed)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = _pick_threshold_for_sensitivity(y_test, y_prob, target_sensitivity=args.target_sensitivity)
    result = evaluate_at_threshold(y_test, y_prob, threshold)

    if len(np.unique(y_test)) > 1:
        save_roc_curve(y_test.astype(int), y_prob, out_dir / "roc_test.png")
        save_pr_curve(y_test.astype(int), y_prob, out_dir / "pr_test.png")
        save_calibration_plot(y_test.astype(int), y_prob, out_dir / "calibration_test.png")

    report = {
        "mode": "train_eval",
        "model_type": model_type,
        "target_sensitivity": float(args.target_sensitivity),
        "chosen_threshold": float(threshold),
        "feature_cols": feature_cols,
        "n_labeled_subjects_used": int(len(df)),
        "artifacts": {"model_path": str(model_path)},
        "split": {
            "method": "GroupShuffleSplit_internal",
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_subject_ids": df.iloc[train_idx]["subject_id"].tolist(),
            "test_subject_ids": df.iloc[test_idx]["subject_id"].tolist(),
        },
        "metrics_at_threshold": result.__dict__,
        "note": (
            "Internal random split (not the same as split_v1.json used by the deep model). "
            "Prefer `make split` then `make train` so baseline and modern share one split."
        ),
    }
    if model_type == "logistic":
        report["model"] = {
            "type": "LogisticRegression",
            "coef": dict(zip(feature_cols, model.coef_.reshape(-1).tolist())),
            "intercept": float(model.intercept_[0]),
        }
    else:
        report["model"] = {
            "type": "RandomForestClassifier",
            "feature_importances": dict(zip(feature_cols, model.feature_importances_.tolist())),
        }

    (out_dir / "train_eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    df.iloc[test_idx][["subject_id", "label"]].assign(pred_prob=y_prob).to_csv(
        out_dir / "test_predictions.csv", index=False
    )

    rr = args.reports_dir if args.reports_dir is not None else reports_dir_from_config()
    if rr is not None:
        mirror_files(rr, baseline_figure_relatives(out_dir))

    print(f"Wrote: {out_dir / 'train_eval_report.json'}")
    print(f"Wrote: {out_dir / 'test_predictions.csv'}")
    print(f"Wrote: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
