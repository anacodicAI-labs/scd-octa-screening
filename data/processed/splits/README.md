# Patient-level splits

## `split_v1.json` (generated — not committed)

This file lists which `subject_id` values belong to **train**, **val**, and **test**. It is produced by:

```bash
make split
```

Requirements:

- `data/processed/labels.csv` must list enough **unique** labeled subjects (typically **≥ 3**) after optional `gradable` filtering inside `scd_octa.splits`.
- Until this file exists, **`make modern_train`** / **`make modern_eval`** will fail because they pass `--split-json` to the modern pipeline.

## `split_v1.example.json` (template only)

Shows the JSON shape. Copy to `split_v1.json` only if you are hand-authoring splits (not recommended); prefer `make split` for reproducibility.
