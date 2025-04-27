# Tetra Examples
A collection of examples of using Tetra to run workflows on RunPod

## Requirements
- Python 3.9 to 3.12
- RunPod Account ([sign up](https://www.runpod.io/console/signup))
- RunPod API Key ([how-to](https://docs.runpod.io/get-started/api-keys))

## Quick Start with `uv`

1. Install [uv](https://github.com/astral-sh/uv) command
1. Run `make`
1. Insert a RunPod API Key as a value to `RUNPOD_API_KEY` into an `.env` file
1. Run `uv run example/examples.py`

## Quick Start with `pip`

1. Run `make venv`
1. Run `. .venv/bin/activate`
1. Insert a RunPod API Key as a value to `RUNPOD_API_KEY` into an `.env` file
1. Run `python -m examples.example`
