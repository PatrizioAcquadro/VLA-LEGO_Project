PYTHON ?= python

.PHONY: check test package
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check src scripts tests
	$(PYTHON) -m mypy src/vla_lego
	$(PYTHON) scripts/check_docs.py

test:
	$(PYTHON) -m pytest tests -q

package:
	$(PYTHON) scripts/check_install.py
