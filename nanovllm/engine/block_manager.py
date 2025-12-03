from enum import Enum
from collections import deque
import xxhash
import numpy as np
import logging

from nanovllm.engine.sequence import Sequence, SequenceStatus


# Logger for hi-cache operations
logger = logging.getLogger("hi_cache")
if not logger.handlers:
    fh = logging.FileHandler("hi_cache_debug.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.setLevel(logging.DEBUG)
logger.debug("\n" +  "="*50 + "\n")

BlockHash = bytes

class Device(Enum):
    CPU = 'cpu'
    GPU = 'gpu'


class Block:

    def __init__(self, block_id, device: Device = Device.GPU):
        self.block_id = block_id
        self.device = device
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    
    @property
    def block_hash(self) -> BlockHash:
        return self.hash


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1


class HiCacheBlockManager:
    """Block manager with GPU/CPU hierarchical caching support.
    
    This manager supports:
    - Allocating blocks on GPU for active sequences
    - Offloading blocks to CPU when GPU memory is full
    - Loading blocks back to GPU when needed
    - Prefix caching with hash-based deduplication
    """
    
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        # Create blocks: first num_gpu_blocks are GPU blocks, rest are CPU blocks
        self.blocks: list[Block] = []
        for i in range(num_gpu_blocks):
            self.blocks.append(Block(i, Device.GPU))
        for i in range(num_cpu_blocks):
            self.blocks.append(Block(num_gpu_blocks + i, Device.CPU))

        self.free_gpu_block_ids: deque[int] = deque(range(num_gpu_blocks))
        self.free_cpu_block_ids: deque[int] = deque(range(num_gpu_blocks, num_gpu_blocks + num_cpu_blocks))
        
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = dict()
        
        # Track GPU to CPU block mapping for swapped sequences
        self.gpu_to_cpu_mapping: dict[int, int] = dict()

    # ===== Helper Function =====

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_gpu_block(self, block_id: int) -> Block:
        """Allocate a GPU block."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        assert block.device == Device.GPU
        block.reset()
        self.free_gpu_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _allocate_cpu_block(self, block_id: int) -> Block:
        """Allocate a CPU block."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        assert block.device == Device.CPU
        block.reset()
        self.free_cpu_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        """Deallocate a block (GPU or CPU).
        
        Note: We intentionally preserve the hash mapping and block metadata
        for prefix caching. This allows future sequences with the same prefix
        to reuse cached blocks.
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count is {block.ref_count}, cannot free"
        
        # Note: We do NOT remove from hash mapping - this enables prefix caching
        # The block's hash and token_ids are also preserved for cache lookup
        
        self.used_block_ids.discard(block_id)
        
        if block.device == Device.GPU:
            self.free_gpu_block_ids.append(block_id)
        else:
            self.free_cpu_block_ids.append(block_id)
            
    # ===== Core API =====
    
    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate enough GPU blocks for the sequence."""
        return len(self.free_gpu_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate GPU blocks for a sequence with prefix caching support."""
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            cached_block_id = self.hash_to_block_id.get(h, -1)
            
            # Check cache hit
            if cached_block_id == -1 or self.blocks[cached_block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Allocate new GPU block
                block_id = self.free_gpu_block_ids[0]
                block = self._allocate_gpu_block(block_id)
            else:
                # Cache hit
                seq.num_cached_tokens += self.block_size
                block_id = cached_block_id
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_gpu_block(block_id)
            
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Deallocate all blocks for a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a new token (may need a new block)."""
        # Need a new block when the current block is full
        needs_new_block = len(seq) % self.block_size == 1
        return len(self.free_gpu_block_ids) >= int(needs_new_block)

    def may_append(self, seq: Sequence):
        """Append a new block if needed, update hash for completed blocks."""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # Current block is full, need a new block
            assert last_block.hash != -1
            block_id = self.free_gpu_block_ids[0]
            self._allocate_gpu_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # Just completed a block, update its hash
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # Block is not yet complete
            assert last_block.hash == -1

    def can_offload(self, seq: Sequence) -> bool:
        """Check if we can offload all GPU blocks to CPU."""
        # Count how many GPU blocks the sequence has
        gpu_block_count = sum(1 for bid in seq.block_table if self.blocks[bid].device == Device.GPU)
        return len(self.free_cpu_block_ids) >= gpu_block_count

    def offload(self, seq: Sequence) -> list[tuple[int, int]]:
        """Offload sequence blocks from GPU to CPU.
        
        Returns:
            List of (gpu_block_id, cpu_block_id) pairs for the actual data transfer.
        """
        assert seq.status == SequenceStatus.RUNNING or seq.status == SequenceStatus.WAITING
        
        transfer_pairs = []
        new_block_table = []
        
        for gpu_block_id in seq.block_table:
            gpu_block = self.blocks[gpu_block_id]
            
            if gpu_block.device != Device.GPU:
                # Already on CPU, keep as is
                new_block_table.append(gpu_block_id)
                continue
            
            # Allocate CPU block
            cpu_block_id = self.free_cpu_block_ids[0]
            cpu_block = self._allocate_cpu_block(cpu_block_id)
            
            # Copy metadata
            cpu_block.hash = gpu_block.hash
            cpu_block.token_ids = gpu_block.token_ids.copy()
            
            # Update hash mapping to point to CPU block
            if gpu_block.hash != -1 and gpu_block.hash in self.hash_to_block_id:
                if self.hash_to_block_id[gpu_block.hash] == gpu_block_id:
                    self.hash_to_block_id[gpu_block.hash] = cpu_block_id
            
            # Record transfer for actual KV cache data movement
            transfer_pairs.append((gpu_block_id, cpu_block_id))
            
            # Deallocate GPU block
            gpu_block.ref_count -= 1
            if gpu_block.ref_count == 0:
                self._deallocate_block(gpu_block_id)
            
            new_block_table.append(cpu_block_id)
        
        seq.block_table = new_block_table
        seq.status = SequenceStatus.SWAPPED

        # Log offload operation
        logger.debug(f"Offload seq_id={getattr(seq, 'seq_id', None)} transfers={transfer_pairs}")

        return transfer_pairs
    
    def can_load(self, seq: Sequence) -> bool:
        """Check if we can load all CPU blocks back to GPU."""
        # Count how many CPU blocks the sequence has
        cpu_block_count = sum(1 for bid in seq.block_table if self.blocks[bid].device == Device.CPU)
        return len(self.free_gpu_block_ids) >= cpu_block_count
    
    def load(self, seq: Sequence) -> list[tuple[int, int]]:
        """Load sequence blocks from CPU back to GPU.
        
        Returns:
            List of (cpu_block_id, gpu_block_id) pairs for the actual data transfer.
        """
        assert seq.status == SequenceStatus.SWAPPED
        
        transfer_pairs = []
        new_block_table = []
        
        for cpu_block_id in seq.block_table:
            cpu_block = self.blocks[cpu_block_id]
            
            if cpu_block.device != Device.CPU:
                # Already on GPU, keep as is
                new_block_table.append(cpu_block_id)
                continue
            
            # Allocate GPU block
            gpu_block_id = self.free_gpu_block_ids[0]
            gpu_block = self._allocate_gpu_block(gpu_block_id)
            
            # Copy metadata
            gpu_block.hash = cpu_block.hash
            gpu_block.token_ids = cpu_block.token_ids.copy()
            
            # Update hash mapping to point to GPU block
            if cpu_block.hash != -1 and cpu_block.hash in self.hash_to_block_id:
                if self.hash_to_block_id[cpu_block.hash] == cpu_block_id:
                    self.hash_to_block_id[cpu_block.hash] = gpu_block_id
            
            # Record transfer for actual KV cache data movement
            transfer_pairs.append((cpu_block_id, gpu_block_id))
            
            # Deallocate CPU block
            cpu_block.ref_count -= 1
            if cpu_block.ref_count == 0:
                self._deallocate_block(cpu_block_id)
            
            new_block_table.append(gpu_block_id)
        
        seq.block_table = new_block_table
        seq.status = SequenceStatus.RUNNING

        # Log load operation
        logger.debug(f"Load seq_id={getattr(seq, 'seq_id', None)} transfers={transfer_pairs}")

        return transfer_pairs
    
    @property
    def num_free_gpu_blocks(self) -> int:
        """Number of free GPU blocks."""
        return len(self.free_gpu_block_ids)
    
    @property
    def num_free_cpu_blocks(self) -> int:
        """Number of free CPU blocks."""
        return len(self.free_cpu_block_ids)
