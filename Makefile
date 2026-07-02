# Demand forecasting + cost-aware A/B -- task runner.  Usage: `make <target>`
PYTHON := .venv/bin/python
KAGGLE := .venv/bin/kaggle
COMP   := m5-forecasting-accuracy
RAW    := data/raw

.PHONY: help install data pipeline test

help:
	@echo "targets:"
	@echo "  install    create .venv and install requirements"
	@echo "  data       download the 3 M5 raw files from Kaggle into data/raw"
	@echo "  pipeline   run the three steps end-to-end (~20 min, <5GB RAM)"
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
	$(PYTHON) -m scripts.prepare_data
	$(PYTHON) -m scripts.train_models
	$(PYTHON) -m scripts.ab_test

test:
	$(PYTHON) -m pytest -q
