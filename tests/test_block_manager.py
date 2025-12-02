"""Unit tests for BlockManager and HiCacheBlockManager."""

import pytest
from nanovllm.engine.block_manager import Block, BlockManager, HiCacheBlockManager, Device
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class TestBlock:
    """Tests for the Block class."""
    
    def test_block_init_gpu(self):
        block = Block(0, Device.GPU)
        assert block.block_id == 0
        assert block.device == Device.GPU
        assert block.ref_count == 0
        assert block.hash == -1
        assert block.token_ids == []
    
    def test_block_init_cpu(self):
        block = Block(10, Device.CPU)
        assert block.block_id == 10
        assert block.device == Device.CPU
    
    def test_block_update(self):
        block = Block(0, Device.GPU)
        block.update(12345, [1, 2, 3, 4])
        assert block.hash == 12345
        assert block.token_ids == [1, 2, 3, 4]
    
    def test_block_reset(self):
        block = Block(0, Device.GPU)
        block.update(12345, [1, 2, 3, 4])
        block.reset()
        assert block.ref_count == 1
        assert block.hash == -1
        assert block.token_ids == []


class TestBlockManager:
    """Tests for the basic BlockManager class."""
    
    def test_init(self):
        bm = BlockManager(num_blocks=10, block_size=4)
        assert len(bm.blocks) == 10
        assert len(bm.free_block_ids) == 10
        assert len(bm.used_block_ids) == 0
        assert bm.block_size == 4
    
    def test_compute_hash(self):
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([1, 2, 3, 4])
        h3 = BlockManager.compute_hash([1, 2, 3, 5])
        assert h1 == h2
        assert h1 != h3
    
    def test_compute_hash_with_prefix(self):
        h1 = BlockManager.compute_hash([1, 2, 3, 4], prefix=100)
        h2 = BlockManager.compute_hash([1, 2, 3, 4], prefix=200)
        assert h1 != h2
    
    def test_can_allocate(self):
        bm = BlockManager(num_blocks=10, block_size=4)
        Sequence.block_size = 4
        
        # Create a sequence with 4 tokens -> 1 block needed
        seq = Sequence([1, 2, 3, 4], SamplingParams())
        assert bm.can_allocate(seq) == True
        
        # Create a sequence with 44 tokens -> 11 blocks needed
        seq2 = Sequence(list(range(44)), SamplingParams())
        assert bm.can_allocate(seq2) == False
    
    def test_allocate_and_deallocate(self):
        bm = BlockManager(num_blocks=10, block_size=4)
        Sequence.block_size = 4
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())  # 2 blocks
        assert bm.can_allocate(seq)
        
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert len(bm.free_block_ids) == 8
        assert len(bm.used_block_ids) == 2
        
        bm.deallocate(seq)
        assert len(seq.block_table) == 0
        assert len(bm.free_block_ids) == 10
        assert len(bm.used_block_ids) == 0


class TestHiCacheBlockManager:
    """Tests for HiCacheBlockManager with GPU/CPU hierarchical caching."""
    
    @pytest.fixture
    def block_manager(self):
        """Create a HiCacheBlockManager with 8 GPU blocks and 8 CPU blocks."""
        Sequence.block_size = 4
        return HiCacheBlockManager(num_gpu_blocks=8, num_cpu_blocks=8, block_size=4)
    
    def test_init(self, block_manager):
        bm = block_manager
        assert bm.num_gpu_blocks == 8
        assert bm.num_cpu_blocks == 8
        assert len(bm.blocks) == 16
        assert len(bm.free_gpu_block_ids) == 8
        assert len(bm.free_cpu_block_ids) == 8
        
        # Check device assignment
        for i in range(8):
            assert bm.blocks[i].device == Device.GPU
        for i in range(8, 16):
            assert bm.blocks[i].device == Device.CPU
    
    def test_can_allocate(self, block_manager):
        bm = block_manager
        
        # Sequence needing 2 blocks
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        assert bm.can_allocate(seq) == True
        
        # Sequence needing 10 blocks (more than GPU blocks)
        seq2 = Sequence(list(range(40)), SamplingParams())
        assert bm.can_allocate(seq2) == False
    
    def test_allocate(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())  # 2 blocks
        bm.allocate(seq)
        
        assert len(seq.block_table) == 2
        assert bm.num_free_gpu_blocks == 6
        assert bm.num_free_cpu_blocks == 8
        
        # Verify blocks are GPU blocks
        for block_id in seq.block_table:
            assert bm.blocks[block_id].device == Device.GPU
    
    def test_deallocate(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        bm.allocate(seq)
        bm.deallocate(seq)
        
        assert len(seq.block_table) == 0
        assert bm.num_free_gpu_blocks == 8
        assert bm.num_free_cpu_blocks == 8
    
    def test_can_append(self, block_manager):
        bm = block_manager
        
        # Sequence with 4 tokens - block is full, next append needs new block
        seq = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq)
        seq.append_token(5)  # Now 5 tokens, needs new block on append
        assert bm.can_append(seq) == True
        
        # Fill up GPU blocks with different sequences to avoid cache hits
        for i in range(7):
            # Use different token sequences to ensure new blocks are allocated
            seq2 = Sequence([100 + i*4, 101 + i*4, 102 + i*4, 103 + i*4], SamplingParams())
            bm.allocate(seq2)
        
        # Now no free GPU blocks
        assert bm.num_free_gpu_blocks == 0
        assert bm.can_append(seq) == False
    
    def test_may_append_new_block(self, block_manager):
        bm = block_manager
        
        # Create sequence with exactly 4 tokens (1 full block)
        seq = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq)
        
        # Simulate adding 5th token
        seq.append_token(5)
        
        # Should allocate new block
        initial_blocks = len(seq.block_table)
        bm.may_append(seq)
        assert len(seq.block_table) == initial_blocks + 1
    
    def test_may_append_update_hash(self, block_manager):
        bm = block_manager
        
        # Create sequence with 7 tokens
        seq = Sequence([1, 2, 3, 4, 5, 6, 7], SamplingParams())
        bm.allocate(seq)
        
        # Add 8th token (completes second block)
        seq.append_token(8)
        
        last_block = bm.blocks[seq.block_table[-1]]
        assert last_block.hash == -1  # Not yet hashed
        
        bm.may_append(seq)
        
        assert last_block.hash != -1  # Now hashed
        assert last_block.token_ids == [5, 6, 7, 8]
    
    def test_can_offload(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())  # 2 blocks
        bm.allocate(seq)
        
        assert bm.can_offload(seq) == True
        
        # Use up CPU blocks
        for i in range(8):
            bm._allocate_cpu_block(bm.free_cpu_block_ids[0])
        
        assert bm.can_offload(seq) == False
    
    def test_offload(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        seq.status = SequenceStatus.RUNNING
        bm.allocate(seq)
        
        original_block_ids = seq.block_table.copy()
        transfer_pairs = bm.offload(seq)
        
        # Should return transfer pairs
        assert len(transfer_pairs) == 2
        
        # Sequence should now be SWAPPED
        assert seq.status == SequenceStatus.SWAPPED
        
        # Block table should now contain CPU block IDs
        for block_id in seq.block_table:
            assert bm.blocks[block_id].device == Device.CPU
        
        # GPU blocks should be freed
        assert bm.num_free_gpu_blocks == 8
        
        # CPU blocks should be used
        assert bm.num_free_cpu_blocks == 6
    
    def test_can_load(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        seq.status = SequenceStatus.RUNNING
        bm.allocate(seq)
        bm.offload(seq)
        
        assert bm.can_load(seq) == True
        
        # Use up GPU blocks
        for i in range(8):
            bm._allocate_gpu_block(bm.free_gpu_block_ids[0])
        
        assert bm.can_load(seq) == False
    
    def test_load(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        seq.status = SequenceStatus.RUNNING
        bm.allocate(seq)
        bm.offload(seq)
        
        assert seq.status == SequenceStatus.SWAPPED
        
        transfer_pairs = bm.load(seq)
        
        # Should return transfer pairs
        assert len(transfer_pairs) == 2
        
        # Sequence should now be RUNNING
        assert seq.status == SequenceStatus.RUNNING
        
        # Block table should now contain GPU block IDs
        for block_id in seq.block_table:
            assert bm.blocks[block_id].device == Device.GPU
        
        # CPU blocks should be freed
        assert bm.num_free_cpu_blocks == 8
        
        # GPU blocks should be used
        assert bm.num_free_gpu_blocks == 6
    
    def test_offload_and_load_preserves_data(self, block_manager):
        bm = block_manager
        
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        seq.status = SequenceStatus.RUNNING
        bm.allocate(seq)
        
        # Store original hash values
        original_hashes = [bm.blocks[bid].hash for bid in seq.block_table]
        original_token_ids = [bm.blocks[bid].token_ids.copy() for bid in seq.block_table]
        
        # Offload
        bm.offload(seq)
        
        # Check metadata preserved on CPU
        for i, block_id in enumerate(seq.block_table):
            block = bm.blocks[block_id]
            assert block.hash == original_hashes[i]
            assert block.token_ids == original_token_ids[i]
        
        # Load back
        bm.load(seq)
        
        # Check metadata preserved on GPU
        for i, block_id in enumerate(seq.block_table):
            block = bm.blocks[block_id]
            assert block.hash == original_hashes[i]
            assert block.token_ids == original_token_ids[i]
    
    def test_prefix_caching(self, block_manager):
        bm = block_manager
        
        # First sequence
        seq1 = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams())
        bm.allocate(seq1)
        
        # Deallocate but keep hash mapping
        bm.deallocate(seq1)
        
        # Second sequence with same prefix
        seq2 = Sequence([1, 2, 3, 4, 9, 10, 11, 12], SamplingParams())
        bm.allocate(seq2)
        
        # First block should be cache hit
        assert seq2.num_cached_tokens == 4
    
    def test_multiple_sequences(self, block_manager):
        bm = block_manager
        
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        seq3 = Sequence([9, 10, 11, 12], SamplingParams())
        
        bm.allocate(seq1)
        bm.allocate(seq2)
        bm.allocate(seq3)
        
        assert bm.num_free_gpu_blocks == 5
        
        bm.deallocate(seq2)
        
        assert bm.num_free_gpu_blocks == 6
        
        # Offload seq1
        seq1.status = SequenceStatus.RUNNING
        bm.offload(seq1)
        
        assert bm.num_free_gpu_blocks == 7
        assert bm.num_free_cpu_blocks == 7


class TestHiCacheBlockManagerEdgeCases:
    """Edge case tests for HiCacheBlockManager."""
    
    @pytest.fixture
    def small_block_manager(self):
        """Create a small HiCacheBlockManager for edge case testing."""
        Sequence.block_size = 4
        return HiCacheBlockManager(num_gpu_blocks=2, num_cpu_blocks=2, block_size=4)
    
    def test_allocate_all_gpu_blocks(self, small_block_manager):
        bm = small_block_manager
        
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        
        bm.allocate(seq1)
        bm.allocate(seq2)
        
        assert bm.num_free_gpu_blocks == 0
        
        # Cannot allocate more
        seq3 = Sequence([9, 10, 11, 12], SamplingParams())
        assert bm.can_allocate(seq3) == False
    
    def test_offload_when_no_cpu_blocks(self, small_block_manager):
        bm = small_block_manager
        
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        seq1.status = SequenceStatus.RUNNING
        seq2.status = SequenceStatus.RUNNING
        
        bm.allocate(seq1)
        bm.allocate(seq2)
        
        # Offload both
        bm.offload(seq1)
        bm.offload(seq2)
        
        assert bm.num_free_cpu_blocks == 0
        
        # Allocate new sequence
        seq3 = Sequence([9, 10, 11, 12], SamplingParams())
        seq3.status = SequenceStatus.RUNNING
        bm.allocate(seq3)
        
        # Cannot offload - no CPU blocks
        assert bm.can_offload(seq3) == False
    
    def test_load_when_no_gpu_blocks(self, small_block_manager):
        bm = small_block_manager
        
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        seq1.status = SequenceStatus.RUNNING
        bm.allocate(seq1)
        bm.offload(seq1)
        
        # Use up all GPU blocks
        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        seq3 = Sequence([9, 10, 11, 12], SamplingParams())
        bm.allocate(seq2)
        bm.allocate(seq3)
        
        # Cannot load seq1 - no GPU blocks
        assert bm.can_load(seq1) == False
    
    def test_empty_sequence(self, small_block_manager):
        """Test handling of minimal sequence."""
        bm = small_block_manager
        
        seq = Sequence([1], SamplingParams())  # 1 token -> 1 block
        bm.allocate(seq)
        
        assert len(seq.block_table) == 1
        assert bm.num_free_gpu_blocks == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
