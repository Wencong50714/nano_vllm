This repo is forked from [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## Quick Start

1) Install the dependencies
```bash
uv venv .venv
source .venv/bin/activate

uv pip install torch
uv pip install -e . --no-build-isolation
```

2) Download model and change model_path in `bench.py` and `example.py`

3) Run the text-only llm
```bash
# Bench
python -m bench

# Single request demo
python -m example
```

## Benchmark Result

| Version | Output Toks | Time (s) | Throughput (token/s) |
| -       | -           | -        | -                    |
| Original| 133,966     | 9.11 s   | 14705.28           |    

## Roadmap

- [] Add Qwen2.5-vl Model suport
- [] Add new feature to support VLM serving