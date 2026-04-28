from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PatientSplit:
    seed: int
    train_subject_ids: list[str]
    val_subject_ids: list[str]
    test_subject_ids: list[str]

    def to_json_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "train_subject_ids": list(self.train_subject_ids),
            "val_subject_ids": list(self.val_subject_ids),
            "test_subject_ids": list(self.test_subject_ids),
        }

    @staticmethod
    def from_json_dict(d: dict) -> "PatientSplit":
        return PatientSplit(
            seed=int(d["seed"]),
            train_subject_ids=[str(x) for x in d["train_subject_ids"]],
            val_subject_ids=[str(x) for x in d["val_subject_ids"]],
            test_subject_ids=[str(x) for x in d["test_subject_ids"]],
        )


def _ensure_unique(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def make_patient_split(
    subject_ids: list[str],
    *,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> PatientSplit:
    """
    Patient-level split: a subject_id can appear in only one split.

    We do:
      - split off TEST from ALL
      - split off VAL from remaining TRAIN
    """
    subject_ids = _ensure_unique([str(s) for s in subject_ids])
    if len(subject_ids) < 3:
        raise ValueError("Need at least 3 unique subject_ids to split train/val/test.")

    all_ids = np.array(subject_ids, dtype=object)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(all_ids, groups=all_ids))
    trainval_ids = all_ids[idx_trainval]
    test_ids = all_ids[idx_test]

    # val_size is fraction of trainval, so we keep "val" roughly val_size of remaining
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    idx_train, idx_val = next(gss2.split(trainval_ids, groups=trainval_ids))
    train_ids = trainval_ids[idx_train]
    val_ids = trainval_ids[idx_val]

    return PatientSplit(
        seed=int(seed),
        train_subject_ids=[str(x) for x in train_ids.tolist()],
        val_subject_ids=[str(x) for x in val_ids.tolist()],
        test_subject_ids=[str(x) for x in test_ids.tolist()],
    )


def load_split(path: Path) -> PatientSplit:
    d = json.loads(path.read_text())
    return PatientSplit.from_json_dict(d)


def save_split(split: PatientSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_json_dict(), indent=2) + "\n")


def _read_labels_subject_ids(labels_csv: Path) -> list[str]:
    df = pd.read_csv(labels_csv)
    if "subject_id" not in df.columns:
        raise SystemExit("dataset_index.csv must contain a subject_id column.")
    df["subject_id"] = df["subject_id"].astype(str)

    # If gradable exists, keep only gradable for split-making (optional but useful)
    if "gradable" in df.columns:
        df = df[df["gradable"].astype(int) == 1]

    # If eye-level rows exist, collapse to subject-level by unique subject_id
    return sorted(df["subject_id"].unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a reproducible patient-level split JSON.")
    ap.add_argument("--labels-csv", type=Path, required=True, help="dataset_index.csv with subject_id (and optional gradable).")
    ap.add_argument("--out", type=Path, required=True, help="Output split JSON path, e.g. data/processed/splits/split_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    args = ap.parse_args()

    subject_ids = _read_labels_subject_ids(args.labels_csv)
    split = make_patient_split(subject_ids, seed=args.seed, test_size=args.test_size, val_size=args.val_size)
    save_split(split, args.out)
    print(f"Wrote: {args.out}")
    print(f"n_subjects: train={len(split.train_subject_ids)} val={len(split.val_subject_ids)} test={len(split.test_subject_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

