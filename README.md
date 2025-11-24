This repo is forked from [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm). In the `multi-modal` branch, I add the multi-modal model serving support and test in `Qwen2.5-VL-7B-Instruct` model in both video and image.

## Quick Start

1) Install the dependencies
```bash
uv venv .venv
source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128p
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

## Roadmap for multi-modal support

- [x] Add Qwen2.5-vl Model suport
- [x] Add new feature to support VLM serving
- [x] Add video support
- [ ] Add benchmark code for multi-modal model
- [ ] Implement E-P-D disaggregation



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