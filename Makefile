.PHONY: dev

# Check if 'uv' is installed
ifeq (, $(shell which uv))
$(error "uv is not installed. Please install it before running this Makefile.")
endif

dev:
	uv venv && \
	( \
	. .venv/bin/activate && \
	uv sync --all-groups && \
	echo "Virtual environment created and dependencies installed." && \
	echo "To activate the virtual environment, run: . .venv/bin/activate" \
	)

requirements:
	uv pip compile pyproject.toml > requirements.txt
