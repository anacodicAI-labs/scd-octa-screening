from __future__ import annotations

from enum import Enum


class ClinicalTask(str, Enum):
    """Clinical prediction target. Only screening is implemented in code today."""

    SCREENING = "screening"
    SCR_STAGING = "scr_staging"
    OCT_THINNING = "oct_thinning"


def require_screening_task(task: str) -> ClinicalTask:
    try:
        t = ClinicalTask(task.strip().lower())
    except ValueError:
        raise SystemExit(
            f"Unknown clinical-task={task!r}. "
            f"Valid values: {', '.join(x.value for x in ClinicalTask)}."
        ) from None
    if t != ClinicalTask.SCREENING:
        raise SystemExit(
            f"clinical-task={task!r} is not implemented yet. "
            "This repo trains binary OCTA abnormal-vs-normal screening only. "
            "SCR staging needs multi-class labels + head; OCT thinning needs B-scan inputs — see mass-general/docs/architecture.txt."
        )
    return t
