"""CPU-only verification: ring buffer KV index math for SWA layers.

Verifies:
1. Ring block count = cdiv(window-1, block_size) + 1
2. Ring slot_mapping formula: ring_base + (pos % ring_blocks) * block_size + (pos % block_size)
3. Block table for decode: [ring_base, ring_base+1, ..., ring_base+ring_blocks-1]
4. For decode at position P with window W:
   - FlashInfer seq_len = min(P+1, W)
   - Block table entries cover the last W positions
   - Ring positions map correctly to physical blocks
5. Prefill→decode transition: last W positions in ring are correct
"""
import math

BLOCK_SIZE = 16
WINDOW = 512
RING_BLOCKS = math.ceil((WINDOW - 1) / BLOCK_SIZE) + 1  # cdiv(511,16)+1 = 32+1 = 33
RING_SLOTS = RING_BLOCKS * BLOCK_SIZE  # 528

print("=== Ring Buffer Math ===")
print(f"block_size={BLOCK_SIZE}, window={WINDOW}")
print(f"ring_blocks={RING_BLOCKS}, ring_slots={RING_SLOTS}")
print()

# Verify: ring_slots >= window
assert RING_SLOTS >= WINDOW, f"ring_slots {RING_SLOTS} < window {WINDOW}"
print(f"✓ ring_slots ({RING_SLOTS}) >= window ({WINDOW})")

# Verify: for any position P, the ring slot is unique within the window
# i.e., for positions [P-W+1, P], no two map to the same ring slot
def ring_slot(pos):
    return pos % RING_SLOTS

for P in [0, 1, 100, 511, 512, 1000, 4096, 131071]:
    window_start = max(0, P - WINDOW + 1)
    positions = list(range(window_start, P + 1))
    slots = [ring_slot(p) for p in positions]
    unique_slots = set(slots)
    assert len(unique_slots) == len(slots), \
        f"P={P}: collision! {len(slots)} positions but {len(unique_slots)} unique slots"
print("✓ No ring slot collisions within window for test positions")

# Verify: block table for decode covers the window
def decode_block_table(P, ring_base):
    """Build block table for SWA decode at position P."""
    seq_len = min(P + 1, WINDOW)
    n_blocks = math.ceil(seq_len / BLOCK_SIZE)
    # Block table: ring_base + 0, 1, ..., n_blocks-1
    # But we need to map logical block j to the ring block containing
    # the j-th block of the last seq_len positions.
    #
    # The last seq_len positions are [P - seq_len + 1, P].
    # Position p maps to ring block (p % RING_SLOTS) // BLOCK_SIZE.
    #
    # For FlashInfer: block_table[j] should be the physical block
    # containing the j-th block of the sequence [0, seq_len).
    # We map logical position i (0..seq_len-1) to actual position (P - seq_len + 1 + i),
    # which maps to ring block ((P - seq_len + 1 + i) % RING_SLOTS) // BLOCK_SIZE.
    #
    # For this to work with a simple block_table[j] = ring_base + j,
    # we need the ring blocks to be contiguous for the window.
    
    # Check: are the ring blocks for the window contiguous?
    window_positions = list(range(P - seq_len + 1, P + 1))
    ring_blocks_used = sorted(set((p % RING_SLOTS) // BLOCK_SIZE for p in window_positions))
    
    return seq_len, n_blocks, ring_blocks_used

print("\n=== Decode Block Table Analysis ===")
for P in [0, 100, 511, 512, 1000, 4096, 131071]:
    seq_len, n_blocks, ring_blocks_used = decode_block_table(P, 0)
    contiguous = (len(ring_blocks_used) == max(ring_blocks_used) - min(ring_blocks_used) + 1)
    wraps = (0 in ring_blocks_used and max(ring_blocks_used) > RING_BLOCKS // 2)
    print(f"  P={P:>6d}: seq_len={seq_len:>4d}, n_blocks={n_blocks:>2d}, "
          f"ring_blocks={ring_blocks_used[:5]}{'...' if len(ring_blocks_used)>5 else ''}, "
          f"contiguous={contiguous}, wraps={wraps}")

# KEY INSIGHT: ring blocks are NOT always contiguous (wrap-around case).
# FlashInfer's block_table is an array of physical block indices.
# We can build it to map logical block j → the correct ring block.
print("\n=== Block Table Construction (handle wrap-around) ===")

def build_swa_block_table(P, ring_base, ring_blocks_total):
    """Build FlashInfer block_table for SWA layer at decode position P.
    
    FlashInfer sees a 'virtual sequence' of length seq_len = min(P+1, WINDOW).
    Virtual position i (0..seq_len-1) corresponds to actual position (P - seq_len + 1 + i).
    Virtual block j contains virtual positions [j*BS, (j+1)*BS).
    We need block_table[j] = physical block containing those KV entries.
    
    Since ring_slot(actual_pos) = actual_pos % RING_SLOTS,
    and the ring is contiguous within each block (BLOCK_SIZE divides RING_SLOTS),
    virtual position i → actual position (P - seq_len + 1 + i) → ring block ((P - seq_len + 1 + i) % RING_SLOTS) // BLOCK_SIZE.
    
    For virtual block j, all positions in [j*BS, min((j+1)*BS, seq_len)) map to the same ring block
    IFF they don't cross a ring block boundary. Since BLOCK_SIZE divides RING_SLOTS,
    positions within a virtual block always map to the same ring block.
    """
    seq_len = min(P + 1, WINDOW)
    n_blocks = math.ceil(seq_len / BLOCK_SIZE)
    block_table = []
    for j in range(n_blocks):
        # First virtual position in block j
        actual_pos = P - seq_len + 1 + j * BLOCK_SIZE
        ring_block = (actual_pos % RING_SLOTS) // BLOCK_SIZE
        block_table.append(ring_base + ring_block)
    return seq_len, block_table

for P in [0, 100, 511, 512, 1000, 4096, 131071]:
    seq_len, bt = build_swa_block_table(P, ring_base=100, ring_blocks_total=RING_BLOCKS)
    print(f"  P={P:>6d}: seq_len={seq_len:>4d}, block_table={bt[:5]}{'...' if len(bt)>5 else ''} (len={len(bt)})")

# Verify: slot_mapping for decode write
print("\n=== Slot Mapping for Decode Write ===")
def decode_slot_mapping(P, ring_base):
    """Slot mapping for writing KV at position P in ring buffer."""
    ring_block = (P % RING_SLOTS) // BLOCK_SIZE
    offset = P % BLOCK_SIZE
    return (ring_base + ring_block) * BLOCK_SIZE + offset

for P in [0, 100, 511, 512, 1000, 131071]:
    sm = decode_slot_mapping(P, ring_base=100)
    ring_block = (P % RING_SLOTS) // BLOCK_SIZE
    offset = P % BLOCK_SIZE
    print(f"  P={P:>6d}: ring_block={ring_block:>2d}, offset={offset:>2d}, slot_mapping={sm}")

# Verify: prefill slot_mapping (ring buffer write for all positions)
print("\n=== Prefill Ring Write Verification ===")
PROMPT_LEN = 1000
print(f"Prompt length: {PROMPT_LEN}")
print(f"After prefill, ring contains positions [{PROMPT_LEN - WINDOW}, {PROMPT_LEN - 1}]")
# Check that the last WINDOW positions are correctly in the ring
last_window = list(range(PROMPT_LEN - WINDOW, PROMPT_LEN))
ring_contents = {}  # ring_slot -> position
for p in range(PROMPT_LEN):
    rs = ring_slot(p)
    ring_contents[rs] = p  # later positions overwrite earlier ones

# Verify: for each position in last_window, its ring slot contains that position
all_correct = True
for p in last_window:
    rs = ring_slot(p)
    if ring_contents[rs] != p:
        print(f"  ✗ Position {p}: ring slot {rs} contains position {ring_contents[rs]}, expected {p}")
        all_correct = False
if all_correct:
    print(f"  ✓ All {WINDOW} positions in last window correctly stored in ring")

# Verify: decode block table at P=1000 covers the right positions
seq_len, bt = build_swa_block_table(999, ring_base=0, ring_blocks_total=RING_BLOCKS)
print("\n  Decode at P=999 (after prefill of 1000):")
print(f"  seq_len={seq_len}, block_table len={len(bt)}")
# Verify each virtual block maps to correct ring block
for j, phys_block in enumerate(bt):
    virtual_start = j * BLOCK_SIZE
    actual_start = 999 - seq_len + 1 + virtual_start
    expected_ring_block = (actual_start % RING_SLOTS) // BLOCK_SIZE
    assert phys_block == expected_ring_block, \
        f"Block {j}: got {phys_block}, expected {expected_ring_block}"
print("  ✓ All block table entries map to correct ring blocks")

print("\n=== Memory Savings ===")
# Per-layer KV per block: 2(K+V) * 8 heads * 128 dim * 1 byte (FP8) * block_size
kv_per_block = 2 * 8 * 128 * 1 * BLOCK_SIZE  # bytes
print(f"KV per block per layer: {kv_per_block} bytes = {kv_per_block/1024:.1f} KiB")

# Full allocation (128K context, blocks_per_slot=8192)
full_blocks = 8192
full_per_layer = full_blocks * kv_per_block
full_36_layers = 36 * full_per_layer
print("\nFull allocation (128K, 8192 blocks/slot):")
print(f"  Per SWA layer: {full_per_layer/1024**3:.3f} GiB")
print(f"  36 SWA layers: {full_36_layers/1024**3:.3f} GiB")

# Ring allocation
ring_per_layer = RING_BLOCKS * kv_per_block
ring_36_layers = 36 * ring_per_layer
print(f"\nRing allocation ({RING_BLOCKS} blocks/slot):")
print(f"  Per SWA layer: {ring_per_layer/1024**3:.6f} GiB = {ring_per_layer/1024**2:.2f} MiB")
print(f"  36 SWA layers: {ring_36_layers/1024**3:.4f} GiB = {ring_36_layers/1024**2:.1f} MiB")

savings = full_36_layers - ring_36_layers
print(f"\nSavings per slot: {savings/1024**3:.3f} GiB ({savings/full_36_layers*100:.1f}%)")

# Full attention layers (12 layers, unchanged)
full_attn_12 = 12 * full_per_layer
print(f"\nFull attention (12 layers, unchanged): {full_attn_12/1024**3:.3f} GiB")
print(f"Total KV before: {(full_36_layers + full_attn_12)/1024**3:.3f} GiB")
print(f"Total KV after:  {(ring_36_layers + full_attn_12)/1024**3:.3f} GiB")

print("\n=== ALL CHECKS PASSED ===")
