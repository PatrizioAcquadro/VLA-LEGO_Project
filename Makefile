PYTHON ?= python

.PHONY: check
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m mypy sim models tracking
	$(PYTHON) tools/check_docs.py
	$(PYTHON) -m scripts.validate_configs
