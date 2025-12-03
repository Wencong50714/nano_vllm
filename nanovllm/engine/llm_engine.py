import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, bench: bool = False, **kwargs):
        self.bench = bench
        self._reset_bench_stats()
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, self.model_runner)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        
        # Track start time for new sequences (when they first get scheduled)
        if self.bench and is_prefill:
            current_time = perf_counter()
            for seq in seqs:
                if seq.start_time is None:
                    seq.start_time = current_time
        
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # Track first token time
        if self.bench and not is_prefill:
            current_time = perf_counter()
            for seq in seqs:
                if seq.first_token_time is None and seq.num_completion_tokens == 0:
                    seq.first_token_time = current_time
        
        self.scheduler.postprocess(seqs, token_ids)

        # Capture first token time (prefill can already emit tokens)
        if self.bench:
            current_time = perf_counter()
            for seq in seqs:
                if seq.first_token_time is None and seq.num_completion_tokens > 0:
                    seq.first_token_time = current_time
        
        # Track finish time and compute metrics
        if self.bench:
            current_time = perf_counter()
            for seq in seqs:
                if seq.is_finished and seq.finish_time is None:
                    seq.finish_time = current_time
                    self._record_bench_stats(seq)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def _reset_bench_stats(self):
        """Reset benchmark accumulators."""
        self.bench_stats = {"ttft": [], "tpot": [], "e2e": []}

    def _record_bench_stats(self, seq: Sequence):
        """Record benchmark statistics for a finished sequence."""
        if seq.start_time is None or seq.finish_time is None:
            return
        
        # E2E latency: total time from start to finish
        e2e_latency = seq.finish_time - seq.start_time
        self.bench_stats["e2e"].append(e2e_latency)
        
        # TTFT: time to first token (only if we generated at least one token)
        if seq.first_token_time is not None and seq.num_completion_tokens > 0:
            ttft = seq.first_token_time - seq.start_time
            self.bench_stats["ttft"].append(ttft)
            
            # TPOT: time per output token (excluding first token)
            if seq.num_completion_tokens > 1:
                decode_time = seq.finish_time - seq.first_token_time
                tpot = decode_time / (seq.num_completion_tokens - 1)
                self.bench_stats["tpot"].append(tpot)
    
    def _print_bench_stats(self):
        """Print average benchmark statistics."""
        if not self.bench:
            return
        
        print("\n" + "="*60)
        print("Benchmark Statistics (Average)")
        print("="*60)
        
        if self.bench_stats["ttft"]:
            avg_ttft = sum(self.bench_stats["ttft"]) / len(self.bench_stats["ttft"])
            print(f"TTFT (Time to First Token): {avg_ttft*1000:.2f} ms")
        else:
            print("TTFT (Time to First Token): N/A")
        
        if self.bench_stats["tpot"]:
            avg_tpot = sum(self.bench_stats["tpot"]) / len(self.bench_stats["tpot"])
            print(f"TPOT (Time Per Output Token): {avg_tpot*1000:.2f} ms")
        else:
            print("TPOT (Time Per Output Token): N/A")
        
        if self.bench_stats["e2e"]:
            avg_e2e = sum(self.bench_stats["e2e"]) / len(self.bench_stats["e2e"])
            print(f"E2E Latency (End-to-End): {avg_e2e*1000:.2f} ms")
        else:
            print("E2E Latency (End-to-End): N/A")
        
        print(f"\nTotal sequences: {len(self.bench_stats['e2e'])}")
        print("="*60)
    
    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if self.bench:
            self._reset_bench_stats()
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        
        # Print benchmark statistics if enabled
        if self.bench:
            self._print_bench_stats()
        
        return outputs
