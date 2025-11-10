This repo is forked from [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## Quick Start

1) Install the dependencies
```bash
uv venv .venv
source .venv/bin/activate

uv pip install torch
uv pip install -e .[all]
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
- [] Implement E-P-D disaggregation

Cur details
- [] Model weight loadder
- [] Qwen3 config initialization inside engine


# Design Doc


## Support dynamic Model load

Inspired by vLLM, this implementation uses a simplified **REGISTRY** mechanism to dynamically load models.

!!! The model name provided in the decorator must match the `hf_config.model_type`. Therefore, it is recommended to run `python -m example.inspect_hf_config`, then copy and paste the model_name into the model class's decorator.

```python
hf_config = AutoConfig.from_pretrained(model_path)

# We've determined the hf_config.model_type is "qwen2", 
# so we add this decorator line above the model class.
@register_model("qwen2")
class Qwen3ForCausalLM(nn.Module):
    pass
```