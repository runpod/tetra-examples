.PHONY: dev

# Ensure Python version is between 3.9 and 3.12
python_version := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

ifeq (,$(filter $(python_version),3.9 3.10 3.11 3.12))
$(error Python version $(python_version) is not supported. Please use Python >= 3.9 and <= 3.12)
endif

dev:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Please install it before running this target."; exit 1; }
	@$(MAKE) update

venv: requirements
	python3 -m venv .venv
	. .venv/bin/activate && \
	pip install pip --upgrade && \
	pip install -r requirements.txt

requirements:
	uv pip compile --upgrade pyproject.toml > requirements.txt

update: requirements
	uv pip sync requirements.txt
