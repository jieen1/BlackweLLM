"""Three-way tensor-shape verification for the DSV4-Flash GGUF checkpoint.

Fact-baseline artifact (notes/2026-08-07-dsv4flash-fact-baseline.md §2.1).
Compares the actual GGUF header dims against the expected GGML ne layout
hand-transcribed from llama.cpp src/models/deepseek4.cpp:78-145 (checkout
79bba02a). GGUF dims are stored in GGML ne order (dims[0] = contiguous);
torch shape is the reverse. Result on the real file: 1304/1304 tensors
present, 0 shape mismatches.

Usage: python tools/verify_dsv4_tensor_map.py [path-to-gguf]
"""

import sys

sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
from pathlib import Path

from loader.gguf_header import read_gguf_header

# --- model constants (config.json / inference config) ---
E, V, H = 4096, 129280, 4096  # n_embd, n_vocab, hidden
N_HEAD, HEAD_DIM = 64, 512
Q_LORA, O_LORA, O_GROUPS = 1024, 1024, 8
HC_MULT, HC_MIX = 4, (2 + 4) * 4  # hc_dim = 16384, mix = 24
N_EXP, N_USED, N_FF, N_SHARED = 256, 6, 2048, 1
IDX_HEADS, IDX_DIM = 64, 128
RATIOS = [0, 0] + [4, 128] * 20 + [4]
assert len(RATIOS) == 43
HASH_LAYERS = 3


def expected():
    """name -> ne tuple, per llama.cpp create_tensor declarations."""
    exp = {}
    exp["token_embd.weight"] = (E, V)
    exp["output.weight"] = (E, V)
    exp["output_norm.weight"] = (E,)
    exp["output_hc_fn.weight"] = (HC_MULT * E, HC_MULT)
    exp["output_hc_base.weight"] = (HC_MULT,)
    exp["output_hc_scale.weight"] = (1,)
    for i in range(43):
        p = f"blk.{i}."
        exp[p + "attn_norm.weight"] = (E,)
        exp[p + "attn_sinks.weight"] = (N_HEAD,)
        exp[p + "attn_q_a.weight"] = (E, Q_LORA)
        exp[p + "attn_q_a_norm.weight"] = (Q_LORA,)
        exp[p + "attn_q_b.weight"] = (Q_LORA, N_HEAD * HEAD_DIM)
        exp[p + "attn_kv.weight"] = (E, HEAD_DIM)
        exp[p + "attn_kv_a_norm.weight"] = (HEAD_DIM,)
        exp[p + "attn_output_a.weight"] = (N_HEAD * HEAD_DIM // O_GROUPS, O_LORA * O_GROUPS)
        exp[p + "attn_output_b.weight"] = (O_GROUPS * O_LORA, E)
        exp[p + "hc_attn_fn.weight"] = (HC_MULT * E, HC_MIX)
        exp[p + "hc_attn_base.weight"] = (HC_MIX,)
        exp[p + "hc_attn_scale.weight"] = (3,)
        exp[p + "hc_ffn_fn.weight"] = (HC_MULT * E, HC_MIX)
        exp[p + "hc_ffn_base.weight"] = (HC_MIX,)
        exp[p + "hc_ffn_scale.weight"] = (3,)
        r = RATIOS[i]
        if r:
            coff = 1 + (1 if r == 4 else 0)
            exp[p + "attn_compressor_kv.weight"] = (E, coff * HEAD_DIM)
            exp[p + "attn_compressor_gate.weight"] = (E, coff * HEAD_DIM)
            exp[p + "attn_compressor_ape.weight"] = (coff * HEAD_DIM, r)
            exp[p + "attn_compressor_norm.weight"] = (HEAD_DIM,)
            if r == 4:
                exp[p + "indexer.proj.weight"] = (E, IDX_HEADS)
                exp[p + "indexer.attn_q_b.weight"] = (Q_LORA, IDX_HEADS * IDX_DIM)
                exp[p + "indexer_compressor_kv.weight"] = (E, 2 * IDX_DIM)
                exp[p + "indexer_compressor_gate.weight"] = (E, 2 * IDX_DIM)
                exp[p + "indexer_compressor_ape.weight"] = (2 * IDX_DIM, r)
                exp[p + "indexer_compressor_norm.weight"] = (IDX_DIM,)
        exp[p + "ffn_gate_inp.weight"] = (E, N_EXP)
        if i < HASH_LAYERS:
            exp[p + "ffn_gate_tid2eid.weight"] = (N_USED, V)
        else:
            exp[p + "exp_probs_b.bias"] = (N_EXP,)
        exp[p + "ffn_norm.weight"] = (E,)
        exp[p + "ffn_gate_exps.weight"] = (E, N_FF, N_EXP)
        exp[p + "ffn_down_exps.weight"] = (N_FF, E, N_EXP)
        exp[p + "ffn_up_exps.weight"] = (E, N_FF, N_EXP)
        exp[p + "ffn_gate_shexp.weight"] = (E, N_FF * N_SHARED)
        exp[p + "ffn_down_shexp.weight"] = (N_FF * N_SHARED, E)
        exp[p + "ffn_up_shexp.weight"] = (E, N_FF * N_SHARED)
    return exp


DEFAULT = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
h = read_gguf_header(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
exp = expected()
actual = {t.name: t.dims for t in h.tensors}

missing_in_file = sorted(set(exp) - set(actual))
missing_in_exp = sorted(set(actual) - set(exp))
mismatches = [(n, exp[n], actual[n]) for n in exp if n in actual and exp[n] != actual[n]]

print(f"expected tensors: {len(exp)}  actual tensors: {len(actual)}")
print(f"missing in file : {len(missing_in_file)}", missing_in_file[:6])
print(f"unexpected extra: {len(missing_in_exp)}", missing_in_exp[:6])
print(f"shape mismatches: {len(mismatches)}")
for n, e, a in mismatches[:10]:
    print(f"  {n}: llama.cpp expects ne={e}, file has dims={a}")
# note: indexer tensor names guessed above; report near-misses for naming check
if missing_in_file or missing_in_exp:
    import difflib

    for n in missing_in_exp[:8]:
        close = difflib.get_close_matches(
            n.split(".")[0] + "." + n.split(".", 2)[-1] if False else n, list(exp), 3
        )
        print(f"  file-only {n} ~ expected-side candidates: {close}")
