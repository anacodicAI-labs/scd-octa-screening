from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .io import build_subject_records, records_to_dataframe


def _count_nonempty(row: dict[str, str], cols: list[str]) -> int:
    return sum(1 for c in cols if row.get(c, "").strip() != "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Index scd-data OCTA TIFF files into a CSV + JSON summary.")
    ap.add_argument("--data-root", type=Path, required=True, help="Path to scd-data directory.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_subject_records(data_root)
    df = records_to_dataframe(records)

    # Add simple completeness counts
    original_cols = ["od_svp", "od_dcp", "os_svp", "os_dcp"]
    bin_cols = ["od_svp_binarized", "od_dcp_binarized", "os_svp_binarized", "os_dcp_binarized"]

    df["n_original"] = df.apply(lambda r: _count_nonempty(r.to_dict(), original_cols), axis=1)
    df["n_binarized"] = df.apply(lambda r: _count_nonempty(r.to_dict(), bin_cols), axis=1)
    df["n_total"] = df["n_original"] + df["n_binarized"]
    df["is_complete_8"] = df["n_total"] == 8

    csv_path = out_dir / "dataset_index.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "data_root": str(data_root.resolve()),
        "n_subjects_with_any_tif": int(df.shape[0]),
        "n_complete_subjects_8": int(df["is_complete_8"].sum()),
        "subjects_complete": df.loc[df["is_complete_8"], "subject_id"].tolist(),
        "subjects_incomplete": df.loc[~df["is_complete_8"], "subject_id"].tolist(),
        "note": "This index only includes subject folders with at least one recognized .tif file.",
    }

    json_path = out_dir / "dataset_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

