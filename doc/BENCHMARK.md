# Benchmark 功能说明

## 修改概述

已成功在代码中添加了 TTFT（Time to First Token）和 TPOT（Time per Output Token）的测量功能。

## 修改的文件

### 1. `nanovllm/engine/sequence.py`
- 在 `Sequence` 类中添加了两个时间戳字段：
  - `first_token_time`: 记录第一个生成 token 的时间
  - `completion_time`: 记录序列完成的时间

### 2. `nanovllm/engine/scheduler.py`
- 添加了 `finished` 列表来存储已完成的序列
- 在 `postprocess` 方法中将完成的序列添加到 `finished` 列表

### 3. `nanovllm/engine/llm_engine.py`
- 在 `generate` 方法中添加了 `benchmark` 参数（默认为 `False`）
- 当 `benchmark=True` 时：
  - 记录每个序列的第一个 token 生成时间
  - 计算所有序列的平均 TTFT 和 TPOT
  - 返回 benchmark 指标字典

### 4. `bench.py`
- 修改为使用 `benchmark=True` 模式调用 `generate`
- 显示详细的 benchmark 结果，包括：
  - 总 token 数
  - 总时间
  - 吞吐量（tok/s）
  - 平均 TTFT（毫秒）
  - 平均 TPOT（毫秒）
  - 序列数量

## 使用方法

### 运行完整 benchmark（256 个序列）：
```bash
python bench.py
```

### 运行简单测试（2 个序列）：
```bash
python test_benchmark.py
```

### 在自己的代码中使用：
```python
from nanovllm import LLM, SamplingParams

llm = LLM("model_path")
prompts = ["Hello", "Hi"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=100)

# 启用 benchmark 模式
outputs, metrics = llm.generate(prompts, sampling_params, benchmark=True)

# 查看指标
print(f"Average TTFT: {metrics['avg_ttft']*1000:.2f} ms")
print(f"Average TPOT: {metrics['avg_tpot']*1000:.2f} ms")
```

## 指标说明

- **TTFT (Time to First Token)**: 从开始生成到第一个输出 token 的时间，主要反映 prefill 阶段的延迟
- **TPOT (Time per Output Token)**: 每个输出 token 的平均生成时间（不包括第一个 token），主要反映 decode 阶段的效率
- **Throughput**: 总体吞吐量，单位是 tokens/second

## 输出示例

```
=== Benchmark Results ===
Total tokens: 142336
Total time: 45.32s
Throughput: 3141.23 tok/s

=== Latency Metrics ===
Average TTFT (Time to First Token): 125.34 ms
Average TPOT (Time per Output Token): 2.15 ms
Number of sequences: 256
```
