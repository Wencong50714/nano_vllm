## 🚀 New Feature

- 2025.12.02: Implement the Basic KV offloading strategy.


## Bench Result

Settings: A100-80G, llm config `llm = LLM(path, enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.2)`
```
Total: 133966tok, Time: 18.05s, Throughput: 7421.63tok/s
Total: 133966tok, Time: 16.10s, Throughput: 8321.31tok/s # use_hicache=True 
```