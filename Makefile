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
	@if [ "$(PY_BIN)" = "" ]; then echo "No virtualenv detected, create one using 'make virtualenv'"; exit 1; fi

.PHONY: install
install: check-venv		## Install the project in dev mode.
	@$(PY_BIN)/pip install -e .[dev,docs,test]

.PHONY: fmt
fmt: check-venv			## Format code using black & isort.
	$(PY_BIN)/isort -v --src src/ tests/ --virtual-env $(PY_ENV)
	$(PY_BIN)/black src/ tests/

.PHONY: lint
lint: check-venv		## Run ruff, black, mypy (optional).
	@$(PY_BIN)/ruff check src/
	@$(PY_BIN)/black --check src/ tests/
	@if [ -x "$(PY_BIN)/mypy" ]; then $(PY_BIN)/mypy project_name/; else echo "mypy not installed, skipping"; fi

.PHONY: test
test: lint			## Run tests and generate coverage report.
	$(PY_BIN)/pytest -v --cov-config .coveragerc --cov=project_name -l --tb=short --maxfail=1 tests/
	$(PY_BIN)/coverage xml
	$(PY_BIN)/coverage html

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

.PHONY: virtualenv
virtualenv:			## Create a virtual environment.
	@echo "creating virtualenv ..."
	@if [ "$(PY_BIN)" != "" ]; then echo "virtualenv already exists, use 'make clean' to remove it."; exit; fi
	@python3 -m venv $(PY_ENV)
	@./$(PY_ENV)/bin/pip install -U pip
	@echo
	@echo "==| Please run 'source $(PY_ENV)/bin/activate' to enable the environment |=="

.PHONY: release
release:			## Create a new tag for release.
	@echo "WARNING: This operation will create a version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@VER_FILE=$$(find src -maxdepth 2 -type f -name 'version.py' | head -n 1)
	@echo "Updating version file :\n $${VER_FILE}"
	@echo __version__ = \""$${TAG}"\" > $${VER_FILE}
	@$(PY_BIN)/gitchangelog > HISTORY.md
	@git add $${VER_FILE} HISTORY.md
	@git commit -m "release: version v$${TAG} 🚀"
	@echo "creating git tag : v$${TAG}"
	@git tag v$${TAG}
	@git push -u origin HEAD --tags

requirements:
	pip install -r requirements.txt

# KERNEL
run_thor_40:
	srun --ntasks 1 --cpus-per-task 4 --mem=32GB --partition=A100 --gpus=3g.40gb:1 -t 24:00:00 --pty bash

run_thor_20:
	srun --ntasks 1 --cpus-per-task 4 --mem=32GB --partition=A100 --gpus=1g.20gb:1 -t 36:00:00 --pty bash

run_baldur:
	srun --ntasks 1 --cpus-per-task 4 --mem=32GB --partition=RTX --gpus=2080ti:1 -t 9:00:00 --pty bash

start_jup:
	jupyter notebook --no-browser --port 40005 --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'

# MODELS (ADDITIONAL)

