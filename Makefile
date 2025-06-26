.PHONY: dev

# Ensure Python version is between 3.9 and 3.13
python_version := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

ifeq (,$(filter $(python_version),3.9 3.10 3.11 3.12 3.13))
$(error Python version $(python_version) is not supported. Please use Python >= 3.9 and < 3.14)
endif

dev:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Please install it before running this target."; exit 1; }
	@$(MAKE) install

install:
	uv sync --all-groups

update: install
	uv lock --upgrade
	uv sync --all-groups
	uv pip compile --upgrade pyproject.toml > requirements.txt
	uv pip sync requirements.txt

venv:
	python3 -m venv .venv

pip: venv
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

clean:
	rm -rf dist build *.egg-info .tetra_resources.pkl
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
