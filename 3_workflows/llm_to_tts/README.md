# LLM to TTS Pipeline

A complete text-to-speech pipeline using Tetra.

## Overview

This example demonstrates how to build an AI pipeline that:
1. **Generates text** using an LLM service (A100 GPU)
2. **Converts text to speech** using a TTS service (RTX 4090 GPU)


## Quick Start

### 1. Install Tetra
```bash
pip install tetra_rp
```

### 2. Set API Key
```bash
export RUNPOD_API_KEY=your_runpod_api_key_here
```

### 3. Run the Pipeline
```bash
python main.py
```
