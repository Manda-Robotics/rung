PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
PYTHON_DIRS = src tests deploy examples

.PHONY: help test coverage lint format-check fmt probes demo examples check build

help:
	@echo "test     offline test suite (no GPU, simulator or API key)"
	@echo "coverage offline tests with terminal, XML and HTML coverage reports"
	@echo "lint     ruff lint and formatting checks"
	@echo "fmt      fix lint and format Python files"
	@echo "probes   embodiment probes against the mock"
	@echo "demo     scripted pick and place on the mock"
	@echo "examples run offline examples and show the remote client/export options"
	@echo "check    lint, test and probes"
	@echo "build    build wheel and source distribution (requires uv)"

test:
	@PYTHONPATH=src $(PY) -m pytest

coverage:
	@PYTHONPATH=src $(PY) -m pytest --cov --cov-report=term-missing --cov-report=xml --cov-report=html

lint: format-check
	@$(RUFF) check $(PYTHON_DIRS)

format-check:
	@$(RUFF) format --check $(PYTHON_DIRS)

fmt:
	@$(RUFF) check --fix $(PYTHON_DIRS)
	@$(RUFF) format $(PYTHON_DIRS)

probes:
	@PYTHONPATH=src $(PY) -m rung.probes --embodiment mock

demo:
	@PYTHONPATH=src $(PY) -m rung.cli --scene banana_in_bowl --policy scripted

examples:
	@PYTHONPATH=src $(PY) examples/mock_episode.py
	@PYTHONPATH=src $(PY) examples/direct_mock.py
	@PYTHONPATH=src $(PY) examples/remote_episode.py --help
	@$(PY) examples/inspect_recording.py --help
	@$(PY) examples/robolab_episode.py --out /tmp/rung-example --dry-run
	@$(PY) examples/read_robolab.py --help

build:
	@uv build

check: lint test probes
