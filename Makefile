# Weekly demand forecasting -- task runner.  Usage: `make <target>`
PYTHON := .venv/bin/python
KAGGLE := .venv/bin/kaggle
COMP   := m5-forecasting-accuracy
RAW    := data/raw

.PHONY: help install data pipeline test

help:
	@echo "targets:"
	@echo "  install    create .venv and install requirements"
	@echo "  data       download the 3 M5 raw files from Kaggle into data/raw"
	@echo "  pipeline   run all phases end-to-end (1 -> 8c)"
	@echo "  test       run the unit tests"

install:
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	mkdir -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f calendar.csv -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f sales_train_evaluation.csv -p $(RAW)
	$(KAGGLE) competitions download -c $(COMP) -f sell_prices.csv -p $(RAW)
	cd $(RAW) && for z in *.zip; do [ -f "$$z" ] && unzip -o "$$z" && rm -f "$$z" || true; done

pipeline:
	$(PYTHON) -m scripts.phase1_data_cleaning
	$(PYTHON) -m scripts.phase2_preprocessing
	$(PYTHON) -m scripts.phase3_eda
	$(PYTHON) -m scripts.phase4_ts_diagnostics
	$(PYTHON) -m scripts.phase5_baselines
	$(PYTHON) -m scripts.phase6_lightgbm
	$(PYTHON) -m scripts.phase7_lstm
	$(PYTHON) -m scripts.phase8_ab_test
	$(PYTHON) -m scripts.phase8b_cost_aware
	$(PYTHON) -m scripts.phase8c_retest

test:
	$(PYTHON) -m pytest -q
