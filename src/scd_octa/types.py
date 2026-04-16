from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

Eye = Literal["OD", "OS"]
Plexus = Literal["SVP", "DCP"]
Kind = Literal["original", "binarized"]


@dataclass(frozen=True)
class OCTAFile:
    subject_id: str
    eye: Eye
    plexus: Plexus
    kind: Kind
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    # Original images
    od_svp: Optional[Path] = None
    od_dcp: Optional[Path] = None
    os_svp: Optional[Path] = None
    os_dcp: Optional[Path] = None
    # Binarized masks
    od_svp_bin: Optional[Path] = None
    od_dcp_bin: Optional[Path] = None
    os_svp_bin: Optional[Path] = None
    os_dcp_bin: Optional[Path] = None

