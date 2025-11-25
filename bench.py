import os
import time
import asyncio
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

model_path = '/data/zbw/huggingface/model_zoo/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/'

async def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    llm = LLM(model_path, enforce_eager=True, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    await llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs, metrics = await llm.generate(prompt_token_ids, sampling_params, use_tqdm=False, benchmark=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {t:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    
    if metrics:
        print(f"\n=== Latency Metrics ===")
        print(f"Average TTFT (Time to First Token): {metrics['avg_ttft']*1000:.2f} ms")
        print(f"Average TPOT (Time per Output Token): {metrics['avg_tpot']*1000:.2f} ms")
        print(f"Number of sequences: {metrics['num_sequences']}")


if __name__ == "__main__":
    asyncio.run(main())
