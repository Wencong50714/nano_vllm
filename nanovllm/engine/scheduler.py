from collections import deque
from typing import Union

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager, HiCacheBlockManager


class Scheduler:

    def __init__(self, config: Config, model_runner=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.use_hicache = config.use_hicache
        self.model_runner = model_runner  # Store reference for data transfer
        
        if self.use_hicache:
            # Calculate CPU blocks based on swap_space_factor
            num_cpu_blocks = config.num_cpu_kvcache_blocks
            if num_cpu_blocks == -1:
                num_cpu_blocks = config.num_kvcache_blocks * config.swap_space_factor
            self.block_manager: Union[BlockManager, HiCacheBlockManager] = HiCacheBlockManager(
                config.num_kvcache_blocks, 
                num_cpu_blocks,
                config.kvcache_block_size
            )
        else:
            self.block_manager: Union[BlockManager, HiCacheBlockManager] = BlockManager(
                config.num_kvcache_blocks, 
                config.kvcache_block_size
            )
        
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()  # New queue for swapped sequences

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Try to load swapped sequences first (if using HiCache)
        if self.use_hicache:
            self._try_load_swapped()
        
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # Try to free up GPU blocks
                if self.use_hicache:
                    # Try to offload a running sequence to CPU
                    if self.running and self._try_offload(self.running.pop()):
                        continue
                    # If we can't offload, preempt the current sequence
                    self.preempt(seq)
                    break
                else:
                    # Original preemption logic
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    
    def _try_load_swapped(self):
        """Try to load swapped sequences back to GPU."""
        while self.swapped:
            seq = self.swapped[0]
            if self.block_manager.can_load(seq):
                # Get transfer pairs and execute data transfer
                transfer_pairs = self.block_manager.load(seq)
                if self.model_runner and transfer_pairs:
                    self.model_runner.execute_load(transfer_pairs)
                self.swapped.popleft()
                self.running.append(seq)
            else:
                break
    
    def _try_offload(self, seq: Sequence) -> bool:
        """Try to offload a sequence to CPU. Returns True if successful."""
        if not isinstance(self.block_manager, HiCacheBlockManager):
            return False
        
        if self.block_manager.can_offload(seq):
            # Get transfer pairs and execute data transfer
            transfer_pairs = self.block_manager.offload(seq)
            if self.model_runner and transfer_pairs:
                self.model_runner.execute_offload(transfer_pairs)
            self.swapped.append(seq)
            return True
        return False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
