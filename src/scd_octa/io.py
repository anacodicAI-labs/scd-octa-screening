from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

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


def build_subject_records(data_root: Path) -> list[SubjectRecord]:
    records: dict[str, SubjectRecord] = {}

    def get_record(sid: str) -> SubjectRecord:
        if sid not in records:
            records[sid] = SubjectRecord(subject_id=sid)
        return records[sid]

    for f in iter_octa_files(data_root):
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

    return [records[k] for k in sorted(records.keys(), key=lambda x: int(x))]


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

