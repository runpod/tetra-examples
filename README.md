# Tetra Examples
A collection of examples of using Tetra to run workflows on RunPod

## Requirements
- Python 3.9 to 3.12
- RunPod Account ([sign up](https://www.runpod.io/console/signup))
- RunPod API Key ([how-to](https://docs.runpod.io/get-started/api-keys))
- UV Python Package Manager ([how-to](https://docs.astral.sh/uv/guides/install-python/)) *(optional)*

## Quick Start with `uv`

1. Run `make`
2. Insert a RunPod API Key as a value to `RUNPOD_API_KEY` into an `.env` file
3. Run `uv run examples/hello_world.py`

## Quick Start with `pip`

1. Run `make pip`
1. Activate Shell `source .venv/bin/activate`
1. Insert a RunPod API Key as a value to `RUNPOD_API_KEY` into an `.env` file
1. Run `python -m examples.hello_world`
