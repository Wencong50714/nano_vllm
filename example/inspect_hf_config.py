from transformers import AutoConfig

# Add your custom model paths here
models = [
    "/home/czh/model_zoo/Qwen2.5-VL-7B-Instruct",
    "/home/czh/huggingface/Qwen3-0.6B/",
]

for model_path in models:
    hf_config = AutoConfig.from_pretrained(model_path)
    # print(hf_config)
    print(hf_config)