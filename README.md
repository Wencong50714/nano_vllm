## 🚀 New Feature

- 2025.12.03: ✅ **Completed asynchronous KV cache offloading implementation**
- 2025.12.03: ✅ **Completed synchronous KV cache offloading implementation**

## Design Doc

### KV Cache Offloading (异步版本)

**核心功能：**
1. ✅ GPU/CPU 双层 KV cache 管理
2. ✅ 异步数据传输 (GPU ↔ CPU)
3. ✅ 自动 offload/load 调度
4. ✅ 与 prefix caching 兼容

**实现说明：**
- CPU 内存使用 pinned memory 加速传输
- 使用 CUDA Stream 实现计算与传输重叠
- Scheduler 自动管理 offload/load 时机
- 支持配置 CPU/GPU 内存比例

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


## Bench Result
`llm = LLM(path, enforce_eager=False, max_model_len=4096, use_hicache=True, gpu_memory_utilization=0.2, bench=True)`

Base Version
```
============================================================
Benchmark Statistics (Average)
============================================================
TTFT (Time to First Token): 93.79 ms
TPOT (Time Per Output Token): 15.22 ms
E2E Latency (End-to-End): 7747.24 ms

Total sequences: 256
============================================================
Total: 133966tok, Time: 18.02s, Throughput: 7434.92tok/s
```


Sync Load version
```
============================================================
Benchmark Statistics (Average)
============================================================
TTFT (Time to First Token): 96.07 ms
TPOT (Time Per Output Token): 16.57 ms
E2E Latency (End-to-End): 8298.03 ms

Total sequences: 256
============================================================
Total: 133966tok, Time: 19.02s, Throughput: 7044.26tok/s
```


Async Load version
```
============================================================
Benchmark Statistics (Average)
============================================================
TTFT (Time to First Token): 94.36 ms
TPOT (Time Per Output Token): 14.72 ms
E2E Latency (End-to-End): 7365.94 ms

Total sequences: 256
============================================================
Total: 133966tok, Time: 17.35s, Throughput: 7719.53tok/s
```
