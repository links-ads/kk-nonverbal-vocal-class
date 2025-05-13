.ONESHELL:
PY_ENV=.venv
PY_BIN=$(shell python -c "print('$(PY_ENV)/bin') if __import__('pathlib').Path('$(PY_ENV)/bin/pip').exists() else print('')")

.PHONY: help
help:				## This help screen
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: init
init:				## Initialize the project template
	@$(PY_BIN)/python init.py

.PHONY: show
show:				## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(PY_BIN)"
	@$(PY_BIN)/python -V
	@$(PY_BIN)/python -m site

.PHONY: check-venv
check-venv:			## Check if the virtualenv exists.
	@if [ "$(PY_BIN)" = "" ]; then echo "No virtualenv detected, create one first."; exit 1; fi

.PHONY: install
install: check-venv		## Install the project in dev mode.
	@$(PY_BIN)/pip install -e .[dev,docs,test]

.PHONY: fmt
fmt: check-venv			## Format code using black & isort.
	$(PY_BIN)/ruff format -v .

.PHONY: lint
lint: check-venv		## Run ruff, black, mypy (optional).
	@$(PY_BIN)/ruff check .
	@$(PY_BIN)/ruff format --check .
	@if [ -x "$(PY_BIN)/mypy" ]; then $(PY_BIN)/mypy project_name/; else echo "mypy not installed, skipping"; fi

.PHONY: test
test: lint			## Run tests and generate coverage report.
	$(PY_BIN)/pytest --cov-report=xml -o console_output_style=progress

.PHONY: clean
clean:				## Clean unused files (VENV=true to also remove the virtualenv).
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@if [ "$(VENV)" != "" ]; then echo "Removing virtualenv..."; rm -rf $(PY_ENV); fi
