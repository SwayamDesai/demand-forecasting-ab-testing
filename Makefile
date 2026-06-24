# Demand Forecasting + Champion-Challenger A/B  --  task runner
# Usage: `make <target>`. Everything runs inside the project venv.

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
KAGGLE := .venv/bin/kaggle
COMP   := m5-forecasting-accuracy
RAW    := data/raw

.PHONY: help install data sample features train experiment test clean

help:
	@echo "targets:"
	@echo "  install     install all deps from requirements.txt"
	@echo "  data        download M5 raw files from Kaggle into data/raw"
	@echo "  sample      carve the fast dev subset (Phase 1)"
	@echo "  features    build lag/rolling/calendar features (Phase 2)"
	@echo "  train       train baselines + deep model (Phase 2/3)"
	@echo "  experiment  run the A/B experiment + stats (Phase 4)"
	@echo "  test        run pytest"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Download only the 3 files we use (~325 MB). Kaggle ships them zipped; unzip + clean up.
data:
	mkdir -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f calendar.csv -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f sales_train_evaluation.csv -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f sell_prices.csv -p $(RAW)
	cd $(RAW) && for z in *.zip; do [ -f "$$z" ] && unzip -o "$$z" && rm -f "$$z" || true; done
	@echo "raw data ready in $(RAW)"

sample:
	$(PYTHON) -m src.data --make-sample

features:
	$(PYTHON) -m src.features

train:
	$(PYTHON) -m src.models.baseline

experiment:
	$(PYTHON) -m src.experiment

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/processed/* mlruns
