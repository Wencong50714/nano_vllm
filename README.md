## 🚀 New Feature

- 2025.12.03: ✅ **Completed synchronous KV cache offloading implementation**

## Design Doc

### KV Cache Offloading (同步版本)

**核心功能：**
1. ✅ GPU/CPU 双层 KV cache 管理
2. ✅ 同步数据传输 (GPU ↔ CPU)
3. ✅ 自动 offload/load 调度
4. ✅ 与 prefix caching 兼容

**实现说明：**
- CPU 内存使用 pinned memory 加速传输
- Scheduler 自动管理 offload/load 时机，并强制同步传输
- 支持配置 CPU/GPU 内存比例

**使用示例：**
```python
llm = LLM(
    model_path,
    use_hicache=True,              # 启用 KV cache offloading
    gpu_memory_utilization=0.3,    # 降低以触发 offloading
    swap_space_factor=2            # CPU blocks = GPU blocks × 2
)
```

**详细文档：** 参见 [KV_OFFLOAD_IMPLEMENTATION.md](./KV_OFFLOAD_IMPLEMENTATION.md)


## Bench Result

Settings: A100-80G, llm config `llm = LLM(path, enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.2)`
```
Total: 133966tok, Time: 18.05s, Throughput: 7421.63tok/s
Total: 133966tok, Time: 16.10s, Throughput: 8321.31tok/s # use_hicache=True 
```