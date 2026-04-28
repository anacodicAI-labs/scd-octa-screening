.PHONY: index biomarkers train split modern_train modern_eval modern_eval_external notebook

# Use repo .venv when present so deps (timm, torch, etc.) match requirements.txt.
# Override explicitly: make PYTHON=/path/to/python modern_train
ifndef PYTHON
  ifneq ($(wildcard $(CURDIR)/.venv/bin/python),)
    PYTHON := $(CURDIR)/.venv/bin/python
  else
    PYTHON := python3
  endif
endif

PYTHONPATH=./src
DATA_ROOT=./data/raw/scd-data
RESULTS=./results
RESULTS_BASELINE=./results/baseline
MODELS=./models
LABELS=./results/dataset_index.csv
SPLIT=./data/processed/splits/split_v1.json
OPTION=optionA
FUSION=concat_mlp
EPOCHS=10
BATCH=8
# Baseline sklearn model: logistic | random_forest
MODEL=logistic
# Optional timm backbone override, e.g. efficientnet_b2 densenet121 (empty = option defaults)
ENCODER_TIMM=
EXT_DATA?=./data/raw/scd-data-external
EXT_LABELS?=./data/processed/labels_external.csv
EXT_OUT?=./results/modern_external/run1

index:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.make_index --data-root "$(DATA_ROOT)" --out-dir "$(RESULTS)"

biomarkers:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.compute_biomarkers --data-root "$(DATA_ROOT)" --out-dir "$(RESULTS)"

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.train_baseline \
		--biomarkers-csv "$(RESULTS)/biomarkers_vessel_density.csv" \
		--labels-csv "$(LABELS)" \
		--out-dir "$(RESULTS_BASELINE)" \
		--models-dir "$(MODELS)" \
		--model-type "$(MODEL)" \
		$(if $(wildcard $(SPLIT)),--split-json "$(SPLIT)",) \
		--target-sensitivity 0.95

split:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.splits --labels-csv "$(LABELS)" --out "$(SPLIT)" --seed 42 --test-size 0.2 --val-size 0.2

modern_train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.modern_model.train \
		--data-root "$(DATA_ROOT)" \
		--labels-csv "$(LABELS)" \
		--split-json "$(SPLIT)" \
		--out-dir "$(RESULTS)/modern/$(OPTION)" \
		--option $(OPTION) \
		--fusion $(FUSION) \
		$(if $(strip $(ENCODER_TIMM)),--encoder-timm $(strip $(ENCODER_TIMM)),) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--target-sensitivity 0.95

modern_eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.modern_model.eval \
		--data-root "$(DATA_ROOT)" \
		--labels-csv "$(LABELS)" \
		--split-json "$(SPLIT)" \
		--ckpt "$(RESULTS)/modern/$(OPTION)/best_model.pt" \
		--out-dir "$(RESULTS)/modern/$(OPTION)" \
		--target-sensitivity 0.95

# External cohort: set EXT_DATA, EXT_LABELS, EXT_OUT, OPTION (for ckpt path), or pass CKPT=...
CKPT?=$(RESULTS)/modern/$(OPTION)/best_model.pt
modern_eval_external:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m scd_octa.modern_model.eval_external \
		--external-data-root "$(EXT_DATA)" \
		--external-labels-csv "$(EXT_LABELS)" \
		--ckpt "$(CKPT)" \
		--out-dir "$(EXT_OUT)"

notebook:
	jupyter lab

