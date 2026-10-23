PYTHON ?= python

.PHONY: check test package
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m mypy src/vla_lego
	$(PYTHON) scripts/check_docs.py
	$(PYTHON) scripts/validate_configs.py

test:
	$(PYTHON) -m pytest tests -q

package:
	$(PYTHON) scripts/check_install.py
