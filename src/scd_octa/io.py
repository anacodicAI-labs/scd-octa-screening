from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .types import OCTAFile, SubjectRecord


_FILENAME_RE = re.compile(
    r"^(?P<id>\d+?)_(?P<eye>OD|OS)_(?P<plexus>SVP|DCP)(?P<bin>_binarized|_binaized)?\.tif$",
    re.IGNORECASE,
)


def parse_octa_filename(path: Path) -> Optional[OCTAFile]:
    """
    Parse files like:
      19_OD_SVP.tif
      19_OD_SVP_binarized.tif
      26_OS_DCP_binaized.tif  (typo observed in our data)
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None

    subject_id = m.group("id")
    eye = m.group("eye").upper()
    plexus = m.group("plexus").upper()
    is_bin = m.group("bin") is not None
    kind = "binarized" if is_bin else "original"

    return OCTAFile(
        subject_id=subject_id,
        eye=eye,  # type: ignore[arg-type]
        plexus=plexus,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        path=path,
    )


def iter_octa_files(data_root: Path) -> Iterable[OCTAFile]:
    for subject_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for p in sorted(subject_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".tif":
                continue
            parsed = parse_octa_filename(p)
            if parsed is not None:
                yield parsed


def _accumulate_octa_file(records: dict[str, SubjectRecord], f: OCTAFile) -> None:
    def get_record(sid: str) -> SubjectRecord:
        if sid not in records:
            records[sid] = SubjectRecord(subject_id=sid)
        return records[sid]

    r = get_record(f.subject_id)
    key = (f.eye, f.plexus, f.kind)

    if key == ("OD", "SVP", "original"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "od_svp": f.path})
    elif key == ("OD", "DCP", "original"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "od_dcp": f.path})
    elif key == ("OS", "SVP", "original"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "os_svp": f.path})
    elif key == ("OS", "DCP", "original"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "os_dcp": f.path})
    elif key == ("OD", "SVP", "binarized"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "od_svp_bin": f.path})
    elif key == ("OD", "DCP", "binarized"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "od_dcp_bin": f.path})
    elif key == ("OS", "SVP", "binarized"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "os_svp_bin": f.path})
    elif key == ("OS", "DCP", "binarized"):
        records[f.subject_id] = SubjectRecord(**{**asdict(r), "os_dcp_bin": f.path})


def iter_octa_files_flat(data_root: Path) -> Iterable[OCTAFile]:
    paths: list[Path] = []
    for ext in (".tif", ".tiff"):
        paths.extend(data_root.glob(f"*{ext}"))
    for p in sorted(paths, key=lambda x: x.name.lower()):
        parsed = parse_octa_filename(p)
        if parsed is not None:
            yield parsed


def build_subject_records(data_root: Path) -> list[SubjectRecord]:
    records: dict[str, SubjectRecord] = {}
    for f in iter_octa_files(data_root):
        _accumulate_octa_file(records, f)

    return [records[k] for k in sorted(records.keys(), key=lambda x: int(x))]


def build_subject_records_flat(data_root: Path) -> list[SubjectRecord]:
    """Same as build_subject_records but all `.tif` files live in one directory (no per-subject folders)."""
    records: dict[str, SubjectRecord] = {}
    for f in iter_octa_files_flat(data_root):
        _accumulate_octa_file(records, f)

    return [records[k] for k in sorted(records.keys(), key=lambda x: int(x))]


def build_subject_records_nested_or_flat(data_root: Path) -> list[SubjectRecord]:
    nested = build_subject_records(data_root)
    if nested:
        return nested
    return build_subject_records_flat(data_root)


def _subject_id_str(v: object) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return str(int(v)) if v == int(v) else str(v).strip()
    return str(v).strip()


def _cell_path(row: pd.Series, col: str) -> Optional[Path]:
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    return Path(s)


def subject_records_from_index_dataframe(df: pd.DataFrame) -> dict[str, SubjectRecord]:
    """
    Build subject_id -> SubjectRecord from a dataset_index-style CSV (absolute or relative paths).

    Used when TIFFs live outside ``data_root`` (e.g. flat ``OCTA AI/``) but paths are stored in the index.
    """
    if df.empty or "subject_id" not in df.columns:
        return {}
    path_cols = [
        ("od_svp", "od_svp"),
        ("od_dcp", "od_dcp"),
        ("os_svp", "os_svp"),
        ("os_dcp", "os_dcp"),
        ("od_svp_binarized", "od_svp_bin"),
        ("od_dcp_binarized", "od_dcp_bin"),
        ("os_svp_binarized", "os_svp_bin"),
        ("os_dcp_binarized", "os_dcp_bin"),
    ]
    if not any(c for c, _ in path_cols if c in df.columns):
        return {}

    out: dict[str, SubjectRecord] = {}
    for _, row in df.iterrows():
        sid = _subject_id_str(row["subject_id"])
        if not sid:
            continue
        kwargs: dict[str, object] = {"subject_id": sid}
        for csv_col, attr in path_cols:
            if csv_col not in df.columns:
                continue
            p = _cell_path(row, csv_col)
            if p is not None:
                kwargs[attr] = p
        out[sid] = SubjectRecord(**kwargs)  # type: ignore[arg-type]
    return out


def merge_subject_records(fs: SubjectRecord | None, csv: SubjectRecord | None) -> SubjectRecord | None:
    """Prefer paths from ``csv`` (index file) when set; otherwise keep filesystem paths from ``fs``."""
    if fs is None and csv is None:
        return None
    if fs is None:
        return csv
    if csv is None:
        return fs
    sid = fs.subject_id
    return SubjectRecord(
        subject_id=sid,
        od_svp=csv.od_svp or fs.od_svp,
        od_dcp=csv.od_dcp or fs.od_dcp,
        os_svp=csv.os_svp or fs.os_svp,
        os_dcp=csv.os_dcp or fs.os_dcp,
        od_svp_bin=csv.od_svp_bin or fs.od_svp_bin,
        od_dcp_bin=csv.od_dcp_bin or fs.od_dcp_bin,
        os_svp_bin=csv.os_svp_bin or fs.os_svp_bin,
        os_dcp_bin=csv.os_dcp_bin or fs.os_dcp_bin,
    )


def records_to_dataframe(records: list[SubjectRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "subject_id": r.subject_id,
                "od_svp": str(r.od_svp) if r.od_svp else "",
                "od_dcp": str(r.od_dcp) if r.od_dcp else "",
                "os_svp": str(r.os_svp) if r.os_svp else "",
                "os_dcp": str(r.os_dcp) if r.os_dcp else "",
                "od_svp_binarized": str(r.od_svp_bin) if r.od_svp_bin else "",
                "od_dcp_binarized": str(r.od_dcp_bin) if r.od_dcp_bin else "",
                "os_svp_binarized": str(r.os_svp_bin) if r.os_svp_bin else "",
                "os_dcp_binarized": str(r.os_dcp_bin) if r.os_dcp_bin else "",
            }
        )
    return pd.DataFrame(rows)

